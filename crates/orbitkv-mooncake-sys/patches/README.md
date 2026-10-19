# Pinned TENT copy-status backport

`tcp-copy-status-and-cuda-context.patch` is applied only to a private build-tree
copy of pinned Mooncake `719735896c86b56fabec6cf3e825fb2ea640597a`. Cargo never
modifies that submodule. Input file Git blob hashes must match the exact pinned source before the
zero-context patch is applied with zero fuzz. Application fails closed.
The development `ORBITKV_MOONCAKE_LIB_DIR` override skips compilation; a qualified
runtime must separately bind those bytes to a build of this patch.

The scoped patch consumes already merged upstream fixes:

- [#3476](https://github.com/kvcache-ai/Mooncake/pull/3476), merge
  `964b8cf0c8208a3fa974d1afe7b10e53e8f05070`: propagate every TCP copy status,
  prevent READ error-text/payload-length collision, choose the owning CUDA device.
- [#3973](https://github.com/kvcache-ai/Mooncake/pull/3973), merge
  `bcdde250c509fb0afa496ebeeba8a6198b772325`: driver pointer classification,
  host-to-host memcpy, and CUDA context selection/restoration.

The second backport extracts the existing copy/probe/free behavior and its driver
header; it excludes later RDMA device-synchronization and plugin changes, which
need separate lifetime gates. There is no new transport API or fault policy.
A broader pin update must remove this temporary patch after separately checking
its intervening transport changes. Upstream merge is not local qualification.

A failed copy now reaches the existing native TCP FAILED state, then the C ABI.
OrbitKV's retained native failure boundary remains unchanged: FAILED does not
release caller buffers, admission, or source owners. This backport does not
supply a physical RDMA/CUDA failure fence or authorize timeout reclamation.

Run the deterministic test helper after native compilation:

```bash
python3 crates/orbitkv-transfer/tests/native/run_copy_status.py \
  --native-build /path/to/target/debug/build/orbitkv-mooncake-sys-ID/out/native \
  --runtime /path/to/frozen/cpu/lib \
  --output /external/artifacts/new-copy-control
```

The helper replaces the Platform loader only in its own executable, keeps valid
registered host buffers, and returns a controlled copy error before submission.
It exercises the four real control-plane copy sites, native TCP status and C ABI,
full destination sentinels, normal subsequent transfers and READ error-length
collision. Run the identical helper against the original pinned library as a
negative control. A code/CPU pass does not qualify installed GPU, RDMA, Manager
SSD, adapter crash or physical termination gates.

Review follow-up preserves pointer-query errors using `cuPointerGetAttributes`:
ordinary unregistered host pointers are identified by its documented successful
zero attributes; unexpected errors never select host memcpy. Driver init failure
rejects copying. Device/context acquisition errors restore the exact saved `CUcontext`, including
an originally null context. If exact restoration fails, the worker and admission
remain held and are not returned for reuse. These follow-ups
have controlled CUDA API tests in `tests/native/cuda_copy_errors.cu`; they are
not yet healthy-GPU qualification or an upstream merged change. A focused
upstream follow-up must be prepared separately from this scoped backport.

One checked context guard performs restoration on every exit, including a
device-selection error after CUDA has changed the current context. Combined
device-selection/restoration failure retains the worker, caller scratch and
admission without submitting a copy. Software controls also saturate all 64
retained context-error calls and require the next call to reject admission.

CUDA errors after a memcpy attempt are not physical drain evidence. The copy
owner retains its current stream, stack and callers' RPC staging buffers inline
instead of returning them. Its device-copy admission limits are 64 live plus
retained calls and 1 GiB of logical copy bytes per Platform; these bounds are
separate from Rust's 1024-batch/32 MiB request-descriptor budget and physical
source/query/pool budgets. A fixed diagnostic marks retention. Pre-submit
pointer/context/stream admission errors return normally; actual-copy errors
retain. This trades liveness for ownership safety until a native termination
primitive is independently established. CPU-only TENT is unaffected.

The device-copy/retention bounds are acquired at the copy owner. TCP request or
response strings may already have been allocated before that boundary; this is
not a total peak RPC allocation limit. Unexpected restore failure after a
completed copy retains the worker; the completed stream may have been returned
only after successful synchronization. Copy/synchronization failures keep both
stream and scratch. Focused controls cover aggregate bytes, exact context,
registered host/managed classification and repeated admission release.

`tcp-worker-terminal-owner.patch` consumes only the TCP worker subset of merged
upstream [#4001](https://github.com/kvcache-ai/Mooncake/pull/4001), merge
`2ccf4cb6c23609aa3cb84176c09da8da9e0417f0`. The worker owns its callback and
batch ID before publishing terminal status and never accesses its task after
publication. Exceptions from actual worker copy staging become FAILED instead
of remaining PENDING. The existing failed-state retention remains mandatory.
Permission, merge-accounting and notification-interval changes from that PR
are not imported by this subset. Use `run_tcp_worker_owner.py` for the real CPU
worker/callback control and old-library negative. This does not establish
RDMA/CUDA quiescence or delayed-notification/page reuse safety.


## Unregister errors and retry ownership

`unregister-retry-ownership.patch` adapts the unregister retry and same-base
coordination contract in [upstream #4573](https://github.com/kvcache-ai/Mooncake/pull/4573)
(head `c3ed2fc6365bf71d48ab5de57b3fd55bc6acc54b`) and the failed-MR retention
contract in [#4249](https://github.com/kvcache-ai/Mooncake/pull/4249)
(head `5945e258daea7b9ef7bea172454ef082ac13cfa2`) to the fixed pin. Both PRs
were open at inspection; this is a local candidate subset, not a merged upstream
fix or a pin upgrade. Existing upstream work is reused without duplicate issues
or PRs.

The engine reserves all base addresses for registration/unregistration. Before
withdrawing the last reference from its local snapshot, it establishes a retry
owner containing the original descriptor and pending transports. Removal errors
reach the existing C ABI and Rust registration token. Successful transports are
removed from the pending set exactly once. A failed metadata publication keeps
a tombstone even when physical removal succeeded. A registration at that base
is rejected until the old unregister and publication finish. Normal batch removal
still publishes once; failure reports the index whose suffix must be retried.
Completed prefixes retain their tombstones until a successful publication covers
their withdrawal revision. The metadata owner captures each snapshot and revision
together before publishing; an older in-flight publication cannot acknowledge a
newer withdrawal. Wildcard batch entries resolve their logical references before
removal, so publication-only tombstones cannot shadow later references.

At the RDMA owner, a refused `ibv_dereg_mr` retains the MR in its context. A
partially successful buffer removal retains only unfinished MRs and reports an
error. The retry uses original keys and does not deregister successful rails again.

This subset does not repair partial registration rollback, arbitrary overlapping
ranges, concurrent engine destruction, failed final destructors or hardware
termination. Callers must keep the engine, registration token and physical backing
alive after an error; an error is not permission to release or reuse memory.
Registration failure/rollback visibility is handled by the separately scoped
repair below. No GPU, RNIC-loss, GDS or physical reclamation support follows from CPU
fault injection.

## Registration errors and rollback ownership

`register-error-ownership.patch` extends the same pinned PR #4573/#4249
contracts. A register call succeeds only when every requested range has a usable
completed transport. Identical new ranges share one physical registration and
retain one reference per occurrence. Existing registrations require identical
memory options; a duplicate cannot silently satisfy a different required
transport.

The existing retirement map reserves new ranges before entering registration
providers. A failed call rolls back its own references and new providers; it
preserves prior owners. Failed publication retains the matched withdrawal
revision. Failed cleanup keeps the original descriptor and unfinished providers
for an explicit retry. Optional fallback remains available only when every range
is usable and incomplete provider cleanup succeeds.

RDMA registration reserves its local buffer entry before registering MRs and
waits for every launched task. Successfully registered rails remain discoverable
even when another rail fails before an RDMA transport is advertised. Removal
retries only surviving MRs. Rust's existing source owner retains the complete
pool and its admission budget when native cleanup is unproven, including a
constructor error with no returned registration token.

Independent code/CPU review accepts production commit `d128c2a7`: nine
registration controls, four MR-owner controls, the existing unregister
expectations, Rust token propagation and six actual Core-owner scenarios
match their declared outcomes. In the no-token case, real native refusal
leaves the complete source pool, engine and charged budget retained. The
matching native build and frozen Rust executables were independently rerun;
the review SHA256 is
`add2b31160d563432b77010c5ba44688d9d61d77cb4157c8dba19972ea938a04`.
Evidence is under
`/root/orbitkv-artifacts/s3-native-registration-repair-20261019/independent-d128-source-cpu-review-1/`.

The native fault controls use test providers, not real RNIC registrations. Warm-up MR
faults, CUDA IPC allocation aliases, device hotplug, consuming unregister/Drop,
terminal destruction and physical termination/reclamation remain unqualified.
