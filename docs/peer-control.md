# Peer control and Mooncake ownership

Block discovery and inventory synchronization now use the
[local global index](distributed-cache.md#local-global-index-and-etcd-metadata).
The custom binary TENT session proposal is retired; its experimental ABI is not
part of OrbitKV runtime.

## Application and transport boundary

| Operation | Implemented path |
| --- | --- |
| Block discovery | Complete local global index; no foreground network query |
| Inventory publication | Background fenced, batched etcd metadata transactions |
| Index synchronization | Fixed-revision etcd snapshot followed by Watch |
| Membership/incarnations | Background etcd registration, lease renewal and Watch |
| Source grants/completion | Bounded batched OrbitKV gRPC |
| Remote payload | TENT READ under source grants |
| P/D payload | TENT WRITE under engine handoff contracts |

Admission, incarnation/generation fences, idempotent completion and retry
ownership belong to OrbitKV. TENT owns registered-memory transfer and native
completion. SEND completion, receiver ACK and drained payload READ are distinct
events. A metadata lease or dead requester cannot prove old READs have drained.

Keep one active READ and at most one authorization lookahead. Submitted work
owns its source guard and destination buffers until native drain. A bounded
completion owner retries release until acknowledged. Index replacement must
preserve these implemented limits and ownership.

SGLang retains engine bootstrap/rooms and routes data through the Rust TENT
adapter. vLLM currently consumes TENT string notifications for P/D. Neither
consumer needs the retired custom binary ABI. Later protocol changes belong in
OrbitKV consumers, preserve page lifetimes and update all callers together.

## Why a native prototype existed

The previous plan chose TENT notifications as a general metadata message bus.
That extra requirement exposed pinned API constraints: C strings/fixed receive
fields cannot carry arbitrary binary frames; native ingress precedes any Rust
queue; RDMA send-slot waits are not bounded application admission; routing and
wakeups needed extra controls. A wrapper could not fix prior native allocations
or truncation.

The isolated prototype therefore changed native framing, queue budgets,
submission, eventfd wakeups and RDMA resource ownership. This followed from that
message-bus choice, not an inability of upstream TENT to transfer KV payloads.
It never served OrbitKV Rust, P/D or production metadata requests.

The selected local-index architecture does not need that bus. Synchronization
and application protocols stay in OrbitKV, etcd uses its supported background
API, and TENT remains the ordinary data plane. The experimental patch is removed
from the application source tree and preserved outside it. No binary-ABI
compatibility layer or alternate runtime is introduced.

Current source gRPC already carries binary protobuf. A different binary transport
does not automatically remove serialization, admission, retries or lifetime
ownership. Measure remaining source-control cost after removing discovery round
trips. Any later transport change needs an end-to-end gain and belongs in OrbitKV
unless a general upstream defect requires a small independent fix.

The two reproduced upstream defects below remain independent transport repairs.
The [integration branch](../crates/orbitkv-mooncake-sys/README.md#native-source)
carries consumed native fixes; OrbitKV cache policy remains in its Rust owners.

## Upstream audit, 2026-09-29

At this audit, OrbitKV pinned Mooncake `719735896c86b56fabec6cf3e825fb2ea640597a`;
the current [integration pin](../crates/orbitkv-mooncake-sys/README.md#native-source)
carries its scoped native fixes. Separately,
latest upstream `main` was checked at
[`be2c57101de5be131729a8b7a8f0c14da61e634c`](https://github.com/kvcache-ai/Mooncake/commit/be2c57101de5be131729a8b7a8f0c14da61e634c).
Existing fixes and open work must be reconciled before updating the native pin;
they are not all unresolved upstream bugs.

| Finding | Upstream status at audit | OrbitKV action |
| --- | --- | --- |
| Retiring notification QPs must remain discoverable until their completions drain | Fixed by [#4062](https://github.com/kvcache-ai/Mooncake/pull/4062) | Use upstream retirement rules when updating the runtime pin |
| Notification QP RTR/RTS attributes must honor endpoint parameters | Fixed by [#4064](https://github.com/kvcache-ai/Mooncake/pull/4064) | Reuse the upstream parameter handling |
| Notifications received through multiple installed transports must all be collected | Fixed by [#4068](https://github.com/kvcache-ai/Mooncake/pull/4068), with send fallback | Use upstream receive aggregation when updating the runtime pin |
| Named notification sends can close a cached peer handle | Open [#4252](https://github.com/kvcache-ai/Mooncake/pull/4252) | Track the existing fix; do not duplicate it |
| C notification result release/reset can leave stale ownership | Open [#4253](https://github.com/kvcache-ai/Mooncake/pull/4253) | Track the existing fix; it does not terminate full-length fields |
| Full-length C notification fields lack a terminating NUL | Reproduced on main; [issue #4380](https://github.com/kvcache-ai/Mooncake/issues/4380), draft [fix #4381](https://github.com/kvcache-ai/Mooncake/pull/4381) | Explicitly terminate both fields in the existing string ABI |
| Terminal endpoint/context destructors ignore unsuccessful native teardown | Reproduced on main; [issue #4382](https://github.com/kvcache-ai/Mooncake/issues/4382), draft [fix #4383](https://github.com/kvcache-ai/Mooncake/pull/4383) | Fail-stop before member destruction if native resources still cannot be released; explicit cleanup remains retryable |

The TCP RPC coroutine already owns its request and server address by value in
both the pinned release and inspected main. Moving arguments into that coroutine
can remove copies; it is not evidence of an upstream borrowed-request lifetime
bug. Binary framing, queue budgets and application sessions were extra prototype
requirements, not distributed-cache prerequisites fixed by these PRs.

Both new fixes were independently built from
`1ad008bc41c6de7c74e02fb1d48713309b880c0d`; the subsequent main commit only changes
CI build parallelism. The string regression fails on unchanged code with glibc
allocation poisoning and passes with the fix (7 local-notification tests).
The teardown regression fails all three terminal-destructor death cases on
unchanged code; the fix passes 29 endpoint tests and 104 RDMA transport tests,
with 2 hardware-dependent skips. All 29 endpoint tests also pass a full native
ASan/UBSan build with leak detection. These are CPU fault-injection results,
not RNIC/provider qualification. Both PRs are drafts pending human review;
local passing tests do not establish upstream CI success or merge status.

OrbitKV's runtime pin and application protocol are unchanged by this audit.
The custom binary experiment is isolated and outside the selected delivery plan.
