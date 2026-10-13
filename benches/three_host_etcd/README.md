# Three-host etcd process-fault harness

This package is the reusable controller and remote-agent implementation for the
S2 three-host etcd process-fault qualification. It tests process/leader/quorum,
Watch/rebuild, membership fencing and final cache-query correctness. It does not
claim independent power, host or network failure domains.

Do not run these source files directly against an old evidence directory. Create
a fresh external stage root with an `inputs/` directory containing every imported Python
module in this package plus `settings.json`. Generate `inputs/SHA256SUMS`, obtain
an independent review of its exact digest, and pass that digest explicitly:

```bash
python3 inputs/campaign.py \
  --root /absolute/external/stage \
  --approved-input-manifest-sha256 <reviewed-input-sha256> \
  --launch-authorization /absolute/external/stage/LAUNCH-AUTHORIZATION.json \
  --approved-launch-authorization-sha256 <authorization-sha256>
```

The controller verifies that digest locally and on every host before launching
an agent. The staged runtime projection and native binaries have independent
hashes in `settings.json`. `resource_lock_agent.py` holds one host-local `flock`
from preflight through postflight cleanup. Existing external-owner status is read
dynamically and bound to its current PID/start ticks; GPU/process/port checks do
not rely on a historical PID. The lock has no timer-based automatic expiry: uncertain
cleanup retains exclusion until an explicit stop record. Every phase boundary
rechecks each lock PID and start tick.

`settings.json` also freezes the process-local direct-network policy. The
controller, every remote launcher, every etcd child, the A100 Manager/pressure
children and all campaign HTTP sessions remove inherited `HTTP_PROXY`,
`HTTPS_PROXY`, `ALL_PROXY` and lowercase equivalents, then set both `NO_PROXY`
and `no_proxy` to `*`. Launch and child records contain only the resulting
absence/policy evidence, never inherited proxy values. Readiness must exercise
the real etcd start, self-version and client/peer health paths under a fake proxy
environment before a formal campaign is reviewed.

All failure cleanup shares one cumulative deadline. A timed-out command or child
is quarantined without SIGKILL, with its PID/start ticks and owner retained. An
unrequested exit is never treated as a planned stop, even when its exit code is
zero. The etcd agent separately records an exact `-SIGTERM` caused by its own stop
request.

The campaign requires exact listener PID, member/endpoint/lease/namespace
identities, monotonic etcd revisions, persistent format identity and strictly
advancing epoch/new-incarnation pairs after restart. The A100 agent preserves
full key/header/revision/lease evidence before a fault, and the controller binds
those records and their three-endpoint absence proofs to its own snapshots. A
dedicated real Manager
source first completes a cache read with source authorization and remote bytes;
after its lease is revoked, the same hash must miss without increasing that
present counter. Required query/source drain metrics must exist and equal zero.
The final DB/WAL, process identities, logs, keys, oracles and postflight remain in
the external stage.

The cold observer is held live while the controller snapshots its exact member;
its configured peer endpoint and owned PID/start ticks are recorded before a
separate normal stop and three-endpoint member-absence proof.

`Campaign.run()` also requires a separately hashed launch-authorization record
that binds the exact input manifest, source commit/tree and an `ACCEPTED`
independent review. Authorization, review and verdict paths are resolved before
stage containment is checked, so `..` and symlink escapes fail closed. A disabled
freeze cannot be executed by calling the runner directly.

Run the source-only CPU controls with:

```bash
uv run --isolated --no-project \
  --with pytest --with numpy --with 'requests>=2.26.0' \
  pytest -q benches/tests/test_three_host_etcd.py
```

The real-process controls need access to their own `/proc/<pid>` entries. A
sandbox that hides those entries cannot validate the PID/start-tick contract.

The current integration consumes initial query phase-series initialization and
requires all ownership and SSD drain metrics. Missing/nonfinite/negative values
fail; missing metrics are never zero. Configuration validation uses explicit
errors and remains active under optimized Python; source-string appearance is
not a semantic or launch acceptance gate. Source helpers are frozen as inputs.

Manager non-killing cleanup verifies task-owned PID/start ticks, records normal
SIGTERM/wait/reap separately, and retains its owner after timeout. Existing main
resource-fd inheritance is preserved. New cohorts require their own resource
inventory/lock identity and independent launch review. Historical failure and
quarantine roots are immutable; this integration does not authorize their cleanup.

Cleanup requests carry the controller's remaining duration. Remote owners start
local deadlines from that duration and only tighten them; they do not compare
cross-host clocks or renew a cleanup allowance. Controller timeout still fails
closed and retains unresolved ownership. The 180-second budget covers the whole
failure cleanup, not each service layer. CPU controls and runtime readiness are
not formal HA results.

`process_qualification.py` is a separate bounded three-host etcd process-only
cohort. It consumes the same node/network/resource-lock owners, but never starts
Manager, cache or GPU consumers. Its frozen four phases are bootstrap,
leader-loss/rejoin, quorum-loss/recovery and full final put/delete equality.
Endpoint/member/cluster identities and raw HTTP responses are preserved.
Each cohort requires a new exact freeze and independent launch acceptance; its
result does not qualify Manager watch/cache recovery or physical fault domains.

Missing or unreadable lock paths quarantine the owner while preserving its
already-open locked descriptor, even if evidence persistence fails. Complete
Manager drain checks parse raw samples, require all six phase labels and reject
nonfinite or nonzero values before aggregation. Actual installed payload and
helper origins are frozen and rehashed; final normal cleanup consumes the same
controller deadline as failure cleanup. New input contracts require a new freeze.

For process-only cohorts, freeze `<node>.runtime.json` for every node in
`inputs/SHA256SUMS`. Capture it with `integrity.process_runtime_snapshot()` using
the actual interpreter after importing the node/network/lock/controller modules.
This includes the executable, standard library, complete HTTP dependency files,
actual import origins, links and mapped native files. All resource-owning helpers
consume it before launch; missing snapshots, changed files and shadow imports
fail closed. Start the controller with the frozen executable and `-B -E -s`;
remote Python commands use those same flags. These flags disable Python environment
and user-site import injection while retaining frozen script imports.
Run the real-interpreter controls in `benches/tests/test_three_host_process_runtime.py`
alongside the ordinary harness controls before requesting launch review.
