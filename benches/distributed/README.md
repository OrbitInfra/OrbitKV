# Peer-cache and TENT benchmarks

Run commands from the repository root. [Benchmark navigation](../README.md).

### GPU-pointer and full-Manager RDMA profile

`rdma_profile` consumes an external `CONTRACT.json`, its SHA256 input index and
an independent `LAUNCH-ACCEPTED.json` bound to that exact index. The contract
freezes both host boot IDs and complete GPU UUID inventories, source/runtime
hashes, file identities, CPU/NUMA binding, RDMA-only configuration, NIC allowlists,
resource lock paths, exact child commands and the ordered cell/sample matrix.
Copy the complete installed wheel runtime into new private directories on both
hosts. Verify ELF closure, actual TENT loading and native import before launch;
never overwrite a mapped library. Keep `harness/benches` separate from the Python
package so source-only imports cannot shadow the installed extension.

Each host supplies `runtime_inputs` naming the frozen registration-check helper
and guard JSON; both files also belong in `runtime_hashes`. The existing owner
consumes complete directory manifests, absent inputs, protected owners and the
ordered inherited lease FDs after acquisition, before child launch and before
final lease release. Declare every Manager cell's private channel domain in the
guard's `channel_domains`, including its `cell`, `config_path`, root and directory
identities. Runtime checks share a 60-second bound within the 100-second RPC wait.

Child environments are literal frozen dictionaries; `MC_FORCE_TCP` must be
absent. After both registered GPU fixtures own healthy listeners, the driver
requests one runtime-admission checkpoint per host before publishing payloads.
Freeze `python_origins`, `observer_library` and each child's `runtime` executable,
required mappings and observer mode. The checkpoint reuses the installed-gate
Python-origin and native-map inspection, including mapped inode/device identity
and executable-byte checks when overlay device numbers differ. It rejects a
loaded observer in off mode. These startup checks are outside transfer timing;
the descriptive driver's PASS does not imply an observation-overhead PASS.

For GPU-pointer cells, compile the byte oracle on each test host before freezing
its resulting hash. A Manager-only contract uses the existing tensor byte and
sentinel checks in `shared_cache_bytes`; it does not need this oracle library:

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 --shared --cudart shared -Xcompiler=-fPIC \
  -gencode arch=compute_90,code=sm_90 benches/distributed/native/tent_gpu_oracle.cu \
  -o /external/rdma-profile/tent_gpu_oracle.so
python -m benches.distributed.rdma_profile \
  --contract /external/rdma-profile/CONTRACT.json \
  --output /external/rdma-profile/cohort-001
```

GPU cells register one 64 MiB device buffer per endpoint and validate its
physical CUDA UUID and pointer type. The bounded matrix compares a 4 KiB guard,
one packed 32 MiB descriptor, and 8,192 4 KiB descriptors separated by real 4 KiB
gaps, across READ/WRITE, both payload directions and one versus four allowed
NICs. Every sample refills the source with a new deterministic seed, clears the
destination and checks both entire 64 MiB buffers with GPU and CPU oracles.
An intentional one-word corruption must produce exactly one mismatch in both
oracles. Filling, clearing and verification are outside transfer timing.

`binding_to_terminal_ms` includes PyO3 conversion, Rust completion polling and
lazy native peer setup/transfer. It does not identify pure fabric time or
separate QP setup causality. The first transfer follows context initialization
and memory registration; report those separate setup clocks and preserve that
first sample. Subsequent warm-up and measured samples have explicit labels.
Four allowed NICs do not imply four NICs were used. Report physical per-NIC
counter deltas (IB data counters use four-byte units), including their possible
background traffic, rather than assuming striping from configuration.

Manager cells reuse `shared_cache_bytes` and `shared_cache_profile` to measure
source DRAM → TENT READ → consumer DRAM → GPU restoration, at 512 KiB, 8 MiB and
32 MiB. Each restore requires exact remote/H2D counter deltas, complete GPU bytes,
sentinel gaps and acknowledged source release. Consumer DRAM eviction, inventory
barriers and oracles are outside the query/restore clock. The worker's existing
10 ms polling sleep is reported separately; its client time is not pure TENT
latency or model-serving TTFT.

A complete Manager-only matrix contains 36 cells: three independently restarted
runs, both physical directions, all three payload sizes and near-NIC versus
four-NIC allowlists. Reverse the second run's cell order. Retain one cold, three
warmup and thirty measured restores per cell, and analyze each run separately
before comparing run medians. Use a fresh cohort after an invalid campaign;
completed cells from that campaign cannot fill the new matrix.

Select unused service ports outside each host's actual
`/proc/sys/net/ipv4/ip_local_port_range`, using distinct ports for restarted
cells. Readiness requires both successful HTTP health and the HTTP/gRPC listening
socket inodes in the owned process's file descriptors. A foreign listener or an
already exited child cannot satisfy readiness. Preserve the original startup
error before cleanup, and record cleanup failures separately. A fresh cell that
has not launched a payload fixture can normally stop and reap its own Managers
without trying to drain a dead HTTP endpoint; after payload admission, the
complete idle proof remains mandatory.

The outer launcher must persist its admission plan and open logs before starting
the controller. After admission, keep its process handle and logs through actual
wait, including receipt-write failures and pending termination signals. A failed
receipt remains an experiment failure even when native drain completes.

Use independently restarted cells and opposite run orders. Thirty measured
samples support descriptive medians/p95/max, not stable p99 or a production
performance promotion. Stop admission at the first invalid/failed cell and keep
its samples and logs. GPU requester stop requires native unregister/free followed
by actual wait/reap; only then may its matching passive peer stop. Manager/fixture
stops require explicit idle evidence and normal SIGTERM. Children inherit the
resource leases. Control loss, unresolved native work or failed final inspection
retains ownership and quarantine; a timeout cannot release those locks. Successful
postflight requires no descendants, GPU allocations or owned UDS sockets.
An operator releasing an idle service after controller loss must distinguish
normal native shutdown from forced CPU-only owner termination. An orphaned
zombie with no native resources is not an actual wait/reap receipt. Preserve
quarantine if the platform init does not reap it; do not modify a platform
process to turn incomplete cleanup into a passing gate.
An independent resource-readmission review may authorize a new normal-path
cohort after proving the exact historical processes have no GPU, mapping,
device-FD, socket or listener ownership. Hold the real resource locks while
archiving and verifying every known marker before removing it. Unknown or live
ownership stays quarantined. This separate admission never accepts the old
campaign or supplies its missing actual wait/reap receipts.
This component profile does not qualify HA, S3 reclamation, native GDS or model
latency. Use its stage breakdown to choose the next consumed production change.


### Native TENT first READ diagnostic

Run `python -m benches.distributed.tent_stage --bind HOST_ADDRESS --native-lib /frozen/native-lib
--role source` on the source and the same command with `--role consumer` on a
different host. Freeze libraries, the RDMA-only TENT configuration, NIC allowlist,
program and workload before launch; capture each endpoint's stdout/stderr in an
external artifact directory. A controller must keep both stdin pipes open and
parse the JSON `ready` record containing the registered CPU address and endpoint.
Send the consumer a JSON line with `operation: "read"`, a unique `id`, the source
`endpoint` and `address`, and `iterations` between 1 and 1000. Every READ clears
the destination and verifies the entire deterministic payload SHA-256.

The returned stages cover open, batch allocation, submit, terminal-status wait
and accepted free. Buffer clearing and SHA validation are outside `native_read_ms`.
Polling spins for the first millisecond and then waits one millisecond between
polls; these observations include that polling delay. Metadata, QP bootstrap and
worker scheduling can occur lazily after submit and are not separately identified
by this C ABI boundary. Use a fresh consumer process for each first-READ sample.

Send `operation: "stop"` to the consumer after `read_complete`, require `stopped`
and a reaped zero exit, then stop the source after every reader has drained.
Timeout/cancellation never releases an uncompleted batch. Control-pipe loss
retains registered buffers; source protocol errors await explicit source stop.
Do not force-kill an isolated owner and infer DMA completion from that signal.
Check recorded PIDs, RPC listeners, NIC counters and native-library hashes before
accepting cleanup. This benchmark excludes Manager grants/indexes, pinned pools,
GPU destination writes, serving, GDS and S3 crash-reclamation qualification.

For finer diagnosis, `tent_native_trace.cpp` interposes four exported C++ symbols
from Mooncake `719735896c86b56fabec6cf3e825fb2ea640597a` on Linux x86-64 with
libstdc++. Build against that revision's exact `tent/common/status.h`; its
nontrivial `Status` return ABI and mangled symbols are not a public API:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Werror -shared -fPIC \
  -I third-party/mooncake/mooncake-transfer-engine/tent/include \
  benches/distributed/native/tent_native_trace.cpp -ldl -o /external/native-trace/libtent_native_trace.so
```

Verify all four interposed symbols exist in the frozen `libtent_shared.so` before
launch. Set `TENT_NATIVE_TRACE_PATH` to a unique external JSON path on each
endpoint. Bind `TENT_NATIVE_TRACE_PID` before executing the traced program;
start the wrapper without `LD_PRELOAD` and install it only for the final exec:

```bash
TENT_NATIVE_TRACE_PATH=/external/native-trace/consumer.trace.json \
  sh -c 'export TENT_NATIVE_TRACE_PID=$$; \
  export LD_PRELOAD=/external/native-trace/libtent_native_trace.so; exec "$@"' \
  trace-owner /frozen/endpoint-program --endpoint-arguments
```

Freeze the actual executable and arguments rather than the example placeholders.
Only that PID records or writes output. Fork/exec descendants retain ordinary
native forwarding but cannot write their parent's path. A missing or malformed
owner with tracing enabled exits 2 before main. Do not preload import/dlopen
readiness checks with a formal cell's output path. The fixed 4,096-event array records
metadata RPC, endpoint construction, connect and bootstrap RPC; it writes only
after normal process teardown. Per-slot publication uses release/acquire;
teardown seals span admission and rejects unfinished calls or unpublished slots.
An absent symbol, output failure, nonzero exit,
overflow, unsuccessful call or missing required stage invalidates the cell.
Run matched trace-off/on warm-READ overhead controls before deeper diagnosis,
with fresh endpoints and the same frozen payload, order, configuration and native
libraries. Preserve failed controls; do not use instrumented timing as a new
production performance qualification.

After both endpoints have drained and exited zero, reconstruct one consumer:

```bash
python -m benches.distributed.tent_trace \
  --trace /external/native-trace/consumer.trace.json \
  --endpoint-stdout /external/native-trace/consumer.stdout \
  --endpoint-exit-code 0 --role consumer \
  --output /external/native-trace/consumer-summary.json
```

Pass the actual exit code returned by the controller's `wait`, not a value
inferred from the stop record. Use `--role source` with the source's own files to summarize its construction
calls. The analyzer checks the process identity, retained event count, statuses,
clock boundaries and required stages. Its consumer window comes from that
endpoint's first READ. Stage calls overlap: the analyzer reports their interval
union rather than adding durations. Source and consumer monotonic epochs cannot
be compared across hosts. The controller must separately prove reaped zero exits
and absent PIDs; a parsed stop record alone cannot establish that cleanup.

Existing `rpc_server_threads` and `transports.rdma.num_lanes` settings are
previously evaluated experimental candidates, not OrbitKV defaults. Preserve
their rejected results; do not restart those candidates without new evidence.
A private-buffer improvement alone cannot establish model latency.

For the current epoch-checked production runtime, use `benches.distributed.rdma_profile`'s
Manager cells and `benches.distributed.shared_cache_profile`, which consume the installed
Rust client. The older `tent_stage` ctypes probe describes an unpatched upstream
48-byte request; the production request includes source epoch evidence and is
88 bytes on Linux x86-64. Do not give the old probe a production library.

The GPU worker records a `query_window` beside its duration statistics. After
actual normal fixture and Manager waits, link the consumer trace to that cell:

```bash
python -m benches.distributed.tent_trace \
  --trace /external/native-trace/consumer.trace.json \
  --manager-cell /external/manager-profile/cell.json --role consumer \
  --output /external/native-trace/manager-summary.json
```

The reader matches the consumer's host, boot, cell, distinct Manager/fixture
PIDs and process starts, query identity, readiness and actual waits. Client query-prefetch windows
include IPC and polling sleeps; they do not identify native READ terminal/free
boundaries. Per-stage clipped temporal unions can overlap and cannot be added
or compared across hosts. Warm windows can have zero setup calls. Freeze and
independently review the complete runtime and resource ownership, then pass
matched observer-off/on overhead controls before consuming instrumented timings.
Existing payload, remote-byte, H2D-byte and resource-drain oracles remain required.

When changing the C++ recorder, compile and run the CPU writer controls against
the same pinned header. These controls require a C++ compiler and are separate
from the default Python gate:

```bash
g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread \
  -I third-party/mooncake/mooncake-transfer-engine/tent/include \
  benches/tests/distributed/tent_native_trace_writer.cpp -ldl -o /external/native-trace/writer-control
```

Run modes `complete`, `fork`, `exec`, `active` and `unpublished` with separate
`TENT_NATIVE_TRACE_PATH` files and each writer's actual pre-exec owner PID.
The first three must exit zero with 1,024 parent events. The child modes must
also report `actual_child_exit: 0`, recorded by the parent's `waitpid`;
a descendant cannot overwrite the parent's trace. `active`
and `unpublished` must exit 2 and cannot produce acceptable evidence. Also check
actual shared-library preload with an exec descendant, malformed/missing owners,
an intentionally foreign owner and disabled tracing. These synthetic CPU controls
do not measure transport performance.


### Native TENT library comparison

`benches.distributed.tent_compare` consumes an external frozen JSON contract. It records two
hosts and their addresses, an SSH config path, RDMA NICs/GID, remote probe/config
paths, the baseline/candidate/original library directories and SHA256 sets,
controller/probe/config hashes, payload and sample counts, pair ordering,
bootstrap seed/draw count and explicit performance guards. Copy the probe and
configuration to **both** hosts, verify their hashes and ELF dependencies, then
freeze the contract before launching endpoints. Keep libraries in separate
directories; never rebuild or replace a mapped runtime.

```bash
python -m benches.distributed.tent_compare \
  --contract /external/native-comparison/FORMAL-CONTRACT.json \
  --output /external/native-comparison/formal
```

Each cell creates a fresh source and consumer, performs one first READ, separate
warmups and measured READs, and clears/checks the entire destination each time.
Timing covers open, allocation, submit, terminal observation and accepted batch
free; clearing and SHA256 are outside that timer. Notifications run **after**
the READ samples. Send 512 unique names and deterministic 2,048-byte payloads to
wrap the pinned 256-slot ring; receive the complete exact set, reject duplicates
or corrupt messages, free every native result allocation, and poll for 25 ms
after the set is complete. This bounded quiet window is a diagnostic check, not
a proof against arbitrarily delayed duplicates.

The consumer must normally drain and exit zero before source stop. Record actual
native mappings, CPU ticks, physical port data/packet counters (data counters
use four-byte units), zero exits and absent PIDs. Bootstrap matched independent
endpoint pairs, never individual requests within a process. Report first READ,
warm median and descriptive p99 separately, including absolute times and
uncertainty. A readiness smoke is not performance qualification. An invalid
prelaunch input starts no endpoint; preserve its evidence and repair/verify
inputs before a separately frozen launch. The first invalid native cell stops
the cohort without retry or replacement. A completed cohort that misses any
frozen performance guard stays failed.

The probe deliberately holds an exported source on control loss or uncertain
native drain, has no reconnect channel, and ignores SIGTERM/SIGINT. A controller
timeout is not normal cleanup or an authorization to reuse its memory. Private
CPU-buffer results do not qualify Manager grants, registered engine pages,
GPUDirect RDMA, serving latency or requester-crash reclamation. Applying an
already merged upstream repair in this isolated comparison does not change the
production dependency pin or create a supported patched runtime.


## Shared-cache qualification

For shared replicas, use the [shared-cache qualification guide](../../docs/shared-cache-qualification.md).
Its HTTP driver works with either engine on existing deployments. The model-serving
restart gate lives in `python/tests/e2e/test_shared_cache.py`; same-host results do
not qualify physical two-host or RDMA deployment.


### First and repeated cross-host serving restores

Use `benches.distributed.shared_cache_serving` against two dedicated, already-started official
engine replicas and their node-local Managers. Configure matching immutable
model artifacts, dtype, cache pages and computation identity, plus a separately
verified RDMA-only TENT profile. This driver does not launch services or detect
RDMA. The topology owner must freeze inputs, record physical NIC counters and
mapped native libraries, and clean up both hosts after success or failure.

```bash
python -m benches.distributed.shared_cache_serving \
  --engine vllm --model /path/to/immutable-model \
  --source-url http://source:8000 --target-url http://consumer:8000 \
  --source-manager http://source:50056 --target-manager http://consumer:50056 \
  --prompts /var/tmp/serving-inputs/fresh-prompts.json \
  --block-tokens 64 --bytes-per-token 147456 --remote-repeats 10 --warmup-repeats 0 \
  --source-medium dram --output /var/tmp/serving-results/profile.json
```

The byte geometry above is for BF16 Qwen3-8B, TP=1/PP=1 and one dense attention
cache group. Derive it from the selected model and registered layout for other
profiles. With 513/1025-token prompts this gate requires exactly 72/144 MiB of
remote fetch and local H2D restoration for every response, complete cold-text
and input/completion-count equality, acknowledged release and exported ownership
gauges at zero. SSD mode additionally evicts source DRAM and requires the exact
physical SSD prefetch byte delta.

Repeated reads use an idle HBM-only reset and consumer DRAM eviction, preserving
the external prefix identity. The official vLLM test server needs
`VLLM_SERVER_DEV_MODE=1` for `/reset_prefix_cache`; `reset_external=true` changes
identity and must not be used. SGLang uses `/flush_cache`. These preparation and
metadata-fence costs are recorded outside the request timer. Failed responses
and their metrics boundaries remain in the output; an existing result is never
replaced. TTFT means client time to the first nonempty streamed text. This is a
correctness gate with descriptive timings; one first read and correlated repeated
reads do not establish throughput, a tail SLO or a matched backend advantage.


### Independent SSD read-batch comparisons

`benches.distributed.shared_cache_compare` evaluates a complete, separately frozen serial
cohort. The topology owner must retain `FROZEN-CONTRACT.json`, `CAMPAIGN.json`
and every `<cell>-profile.json` together in an external artifact directory.
The comparator rejects incomplete cells, changed profile hashes, mismatched
model/token IDs or output, missing rows, incorrect bytes and a batch setting
that does not change the consumed authorization/READ counts. Compare complete
generated text and native prompt/completion counts; retain process-specific
SGLang usage metadata without requiring UUIDs or timing fields to match.

```bash
python -m benches.distributed.shared_cache_compare \
  --root /var/tmp/ssd-peer-batch/cohort \
  --contract /var/tmp/ssd-peer-batch/cohort/FROZEN-CONTRACT.json \
  --output /var/tmp/ssd-peer-batch/analysis.json
```

The 2026-10-10 component contract freezes five independent, alternating-order
process pairs per engine. Each cell retains five warmup and 20 measured restores
for each 513/1,025-token prefix; warmup exclusion is declared before launch.
Source and consumer use the same two physical H20 hosts, official releases,
immutable Qwen3-8B model, layouts and byte budgets in both variants. Only the
consumer's existing ordinary `--query-read-batch` changes from 32 to 128 MiB.
The source uses io_uring SSD and a 32 MiB read batch throughout. TENT reads into
consumer DRAM, followed by local H2D; this does not measure remote GPU-buffer
RDMA or native GDS. No timer sampler or tracer is enabled; reset, metadata fences,
drain and metrics collection remain outside request timing.

The comparator independently validates each cell's `stops` records; a PASS label
and correct output are insufficient. The frozen `source_host`/`consumer_host`
indices and service names must match. Stop and reap the consumer engine, source
engine, consumer Manager and source Manager in that order. Both Managers remain
available for context unregister. Each record must establish a requested SIGTERM
to a running service, zero leader exit, no forced cleanup, no remaining process
group and released ownership. Engine shutdown logs must be complete and contain
the selected release's completion markers without tracebacks, `EngineDeadError`
or native bootstrap/ignored-exception/SIGQUIT failures. Preserve literal service
exit codes, native child termination and cleanup failures separately. The first
invalid runtime, correctness or cleanup cell stops dependent launches; retain
its profile rows and never analyze the completed subset as a qualified cohort.
Valid performance failures remain in the predeclared comparison.

Compute one measured TTFT/E2E median per process cell and bootstrap the matched
pair ratios, never the correlated requests. The frozen component guard requires
at least a 5% geometric-mean TTFT reduction, a 95% pair-bootstrap upper bound
below parity, and no pair's TTFT or E2E median ratio above 1.05. Report the
intervals and every pair, including performance failures. A zero analysis-process
exit means the analysis completed: inspect its `state` for the component result.
Even `VALID_COMPONENT_PASS` leaves `production_qualified=false`; matched pressure
and cancellation/drain controls precede a production recommendation. This cohort
does not qualify p99, throughput or an advantage over another cache backend.

`benches.serving.single_node` consumes the same native log check. Its service owner sends
normal SIGTERM to the leader so the engine can drain its own workers, then waits
for the leader and live members of its process group. It rejects a mismatched
`/proc` PID namespace before launching any service. A required group SIGKILL,
an unreaped zombie, an early/nonzero exit, an inspection error, a missing/truncated
shutdown log or a native failure prevents `summary.json` from being written.
The Linux owner adopts orphaned workers and reaps only its registered process
group. If inspection fails, it retains the caller's ownership while checking
kernel group existence and terminating/reaping that group; it never treats an
inspection error as an empty group. Cleanup records retain adopted child exits,
inspection failures and
the shutdown byte range and failure; a workload failure stays the primary error.
These Linux process checks do not establish physical CUDA/RDMA lifetime proof.

The retained vLLM consumer shutdown error matches the ordering described in
upstream [issue #48745](https://github.com/vllm-project/vllm/issues/48745): engine
teardown can wake the output handler before its cancellation. The related
[PR #49000](https://github.com/vllm-project/vllm/pull/49000) is open as checked on
2026-10-10. Official vLLM 0.31.0 remains unchanged. Do not suppress this error in
the benchmark or treat a CPU protocol check as a repaired hardware lifecycle.
