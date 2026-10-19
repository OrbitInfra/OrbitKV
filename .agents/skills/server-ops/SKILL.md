---
name: server-ops
description: Configure or diagnose OrbitKV Cache Managers, local process transport, DRAM/SSD budgets, etcd membership, inventory-fed global indexes and peer transfers. Use for deployment and runtime incidents.
---

# OrbitKV Manager operations

Resolve paths from the Git root and read `AGENTS.md`. Use the built
`orbitkv-cache-manager --help` and `docs/server.md` for current flags; do not copy
an old flag list. The source binary is `orbitkv-cache-manager`; the wheel exposes
the same console command and bundles `orbitkv-cache-manager-py`.

Start with `docs/deployment.md` for topology and `docs/p2p.md` for distributed
startup. Every node has an independent Manager. Distributed mode needs a concrete
peer address, etcd endpoints, a unique stable node ID and the same cluster name.
`orbitkv-catalog` is an embedded local index, not a separate service.

Client and Manager CUDA ordinals may differ. Registration joins the tensor's full
GPU UUID to the Manager-visible device and retains that route for Publish/Restore;
use matching lifecycle-protocol-9 artifacts. NUMA selection uses those visible
UUIDs. Known payload placement uses `mbind` before first touch and requires the
selected memory node in the process's allowed nodes. CPU affinity alone cannot
override an inherited interleave memory policy; inspect actual resident pages.

For installed registration qualification, use a stdlib-only `-I -S` control
interpreter with a durable actual-wait owner. Freeze absent import search paths
as well as present files. The embedded Manager's explicit Python environment
must disable user-site and cover consumed system-site `.pth` files. Require its
actual registry Torch/helper origin records and `sys.path`, plus mapped native
library provenance; an independent venv import does not prove Manager loading.
For installed pytest children, use `python/tests/support/installed_pytest.py` with
explicit canonical installation and harness roots. CPU readiness must exercise
that exact frozen entry with both `--probe` and pytest `--collect-only`; a separate
`python -c` import does not validate its path selection or collection boundary.
When an old channel domain is quarantined, unique service names do not isolate
iceoryx2's global counter or stale-node cleanup. Use the same frozen absolute
`ORBITKV_CHANNEL_CONFIG` for client and Manager with a separate private prefix
and canonical root. The Rust channel owner consumes it before Node creation,
independently of Manager cwd; recheck resource scope and READY before launch.
Precreate the private node/service directories without symlinks, freeze each
path-chain identity, and keep the domain immutable under its resource owner
through cleanup. The dependency uses paths after the Rust precheck; concurrent
privileged renames are not atomically fenced by the channel configuration.

For remote metadata scope, omission means `AllNamespaces`. Repeat
`--metadata-namespace` only with complete storage namespaces logged by actual
registration, or use `--metadata-empty-scope` for an explicit empty set. Scope is
startup configuration: change it by restarting and fully bootstrapping the
Manager. Never infer it from model display names, prefixes or query misses. The
allowlist reduces record bytes, not peer-session fanout, and is not tenant auth.

Trace an incident through its owner:

1. Check process health and the engine's UDS/iceoryx2 registration before diagnosing
   a cache miss as a transport failure. See `crates/orbitkv-server/src/endpoint/`.
2. Check `/cache/metadata`: membership validity/revision, explicit coverage,
   scope digest/kind/count, active/staging index bytes, installed views and stream
   queues. Distinguish a complete empty owner view from a missing view. A
   scope-outside remote lookup is unavailable even when local cache service works.
   `/cache/sync` returns a source inventory fence; a controlled requester must
   install it with the same scope via `/cache/metadata/await`. A received frame or
   source head is not that watermark.
3. Check source grants and generation checks in `crates/orbitkv-core/src/peer/`,
   then TENT READ/completion. Index freshness does not authorize an address.
4. Inspect query, source export, SSD staging and GPU admission budgets using
   `docs/metrics.md`. Lease expiry or a deadline cannot establish native drain.

Use native TENT for payloads and OrbitKV source-control RPCs. Do not revive
Catalog placement/lookup RPCs or change etcd's transport to fix a payload problem.
Use test-owned processes for faults; preserve other workloads and collect logs
outside the repo. Keep same-host TCP, physical two-host TCP and RDMA evidence
separate. Build first, then test with native binaries/libraries frozen.

The source-shutdown candidate calls the irreversible native source gate before
unregistering the pinned-pool owner. Only CPU-only/TCP-only handler drain plus
empty Rust requester admission permits release. Unsupported paths or errors
retain engine, tokens and pool in the bounded source registry; do not retry
unregister from token Drop or reset its process admission. Native Delegate is
rejected in this narrow profile because its pointer wire has no source epoch.
This is whole-engine shutdown, not a per-ticket orphan, GPU or RDMA receipt.
Pin/Unpin and queued global-executor handlers require the same ownership:
dispatch is counted before offload and drained before service deletion. Source
close shares its deadline with that drain; server I/O shutdown alone is not
proof that a global-executor callback ended. Readiness is published after all
native worker initialization.
Keep CPU host-registration/C ABI test-shim controls separate from real CUDA
registration and installed hardware gates; see S3 and `docs/transport.md`.

For a normal Manager stop, require SIGTERM to fence membership before gRPC waits
for inventory streams, then drain lifecycle ownership and revoke the member lease.
Do not accept a test helper's SIGKILL fallback as graceful shutdown evidence.
Verify the member key is absent before reusing a node ID, and require a larger
epoch plus a different incarnation after restart. Current/peak inventory session
counters are abort-safe diagnostics: a replaced follower must return current
sessions to the real live count even when its task is aborted.

For repeated-key metadata traffic, `--inventory-stream-coalesce-ms` selects a
0–5 ms quiet window; the default is 0. A requested fence interrupts intentional
wait but still requires complete input-interval installation. Compare inventory
input/output counters and stream encoded bytes with measured observer visibility
and effective hits. Etcd network counters now cover membership/configuration,
not block propagation. A lower mutation count alone does not establish serving
improvement.

For S2.10 performance qualification, keep `s2.10-performance-v2` endpoint names
separate. Serial or concurrent `/cache/metadata/await` duration measures barrier
verification, not ordinary visibility. Ordinary visibility uses the source
`inventory_last_change_mono_ns` and matching owner `installed_mono_ns` without a
sync/flush in the measured path; compare them only on one host after matching the
node epoch, incarnation, scope digest and sequence. Report HTTP observation delay
separately. All/scoped establishes filtering benefit, while save/query isolation
uses the same scope and matched quiet versus metadata-only pressure controls.
Reject pressure runs whose foreground always lands in a quiet phase. Source
cadence, actual publication/operation clocks, advancing observer windows and
sampler overhead must be recorded symmetrically. Short capacity runs and runs
that skip real restore cannot qualify the 16-owner threshold. Formal metadata-only
pressure uses a release-built server-test fixture with the same frozen record
cadence and budgets; preserve dev-profile observations separately. Controllers
must reject optimized Python, validate cadence after each cell, and bound helper
and measurement waits against the fresh global deadline. A watchdog leaves an
active native process isolated; it does not establish drain.

For scoped-stream qualification, pair all-domain and scoped runs with the same
exact namespace/key/payload mutations and coalescing. Report input/output filter
records and CPU, encoded bytes/frames, scope-bound coverage, active/staging/index
bytes, RSS, queue peaks, visibility, bootstrap/repair, and lookup/update lock
wait/hold time. Empty filtered intervals must advance only source-proven coverage.

For inference interference, measure the physical NIC/direction, PCIe/NUMA and
engine collective traffic as well as TENT transfers. Manager query-byte budgets
are not a node-wide network scheduler; engine-local P/D WRITE is a separate
submitter. Receiver credit/pacing is planned in S6, not an existing runtime flag.
Source authority, network credits and destination lifetime need separate drain
evidence. Report that distinction when diagnosing congestion or retained memory.

Pressure qualification must pace after the prior completed source iteration as
well as the original schedule. A delayed tick cannot emit compressed catch-up
bursts. Keep the frozen minimum interval and mutation count; never fix this by
lowering a threshold or dropping events. A terminal exposure failure remains an
invalid run, but should still execute explicit local drain, unregister and
normal Manager shutdown before raising and preserve that cleanup evidence.

For S2.10a stage diagnosis, enable `ORBITKV_TRACE_TRANSFERS=1` and a positive
`ORBITKV_DIAGNOSTIC_TIMELINE_LIMIT` symmetrically in quiet and pressure runs.
The latter is capped at 65,536 events and zero means off. Correlate client,
Manager, insert and SSD stages only by session epoch/token plus channel request
ID on one host monotonic clock. A limit event or missing required stage invalidates
the run. Save return means GPU copy/encode completed and insertion was queued;
insert and SSD completion are later diagnostic stages. Use per-thread schedstat
when perf is unavailable, and report that precision limit instead of inferring
scheduler causality. Diagnostic results never replace the frozen isolation cohort.
With a positive limit, all transfer/stage records share the bounded asynchronous
writer; required stage/query-path entries are fixed-size until that writer
formats them. Require normal Manager shutdown to flush it before accepting linkage.
An instrumentation pilot that fails its frozen overhead guard blocks a diagnostic
matrix even when correlation, correctness and cleanup otherwise pass.
Treat `publish_manager_response_publish` as the before-publication boundary; its
client interval includes publication plus notification. Writer-start failure or
missing post-drain flush invalidates linkage rather than permitting fallback logs.
Shutdown must stop channel admission, await all admitted Publish continuations,
then flush insert/SSD owners before flushing diagnostic output.

Manager worker raw-copy shapes share that bounded timeline. After independently
checking normal shutdown and owned-resource cleanup, use `benches.transfer.raw_copy_shapes`
to require complete flush/count/shape coverage. Bind raw batches to the actual
Manager receipt and require one completion process for a one-Manager gate.
Engine-local asynchronous H2D is outside this trace. CPU submission and host
stream waiting include instrumentation and scheduling; a structural model pass
does not replace matched overhead, device-event or application performance gates.
