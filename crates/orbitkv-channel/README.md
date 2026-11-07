# orbitkv-channel

Versioned request/response channel between inference processes and a Cache Manager
on the same host. The control message is fixed at 64 bytes; KV payloads remain
in registered CUDA IPC pages or Cache Manager storage. The channel does not
select RAM, SSD, or a remote replica: the Cache Manager resolves each cache
request and uses OrbiTE when a remote fetch is available.

The production server owns a thread-safe iceoryx2 service and currently serves
`Ping`, `QueryBundle`, `CancelQuery`, `Publish`, `Restore`, `Release`,
`ObserveCompletion`, and `Shutdown`, including session-epoch fencing.
`ObserveCompletion` accepts bounded, low-cardinality physical completion
evidence; it carries no request ID or state key. A mode-0600 Unix socket
verifies peer credentials and passes sealed descriptor and Restore-grant memfds
plus separate Restore, retirement and Publish-reply eventfds. Every client owns one arena slot guarded by a client
token and a monotonic request/response generation. Python exposes diagnostics
through `ChannelProbeClient` and cache operations through the PyO3
`CacheManagerClient`. Rust `CacheClient` owns query revisions, warming interests,
an independent publish connection and restore handles bound to their issuer.
Restore uses a generation-tagged shared result slot plus the bootstrapped eventfd
with bounded fallback polling. The native client consumes and acknowledges the
result without a terminal Poll RPC. An outcome waiter publishes directly after
GPU drain instead of waiting for a dispatcher scan. Waiting releases the GIL and never blocks the Cache Manager dispatcher;
a deadline does not release GPU destinations. The low-level Rust `ChannelClient`
owns descriptor framing and session failure, without a second Python facade.

The client reserves the operation identity before sending Restore. Manager
claim and client cancellation compete atomically before lease consumption;
a claimed operation keeps its handle even if its submission ACK is lost.
Preparation failures are terminal Failed results on that handle. Only the
descriptor channel closes after an ambiguous call: mapped results remain
readable, and UDS closure is not a GPU completion fence. The fixed ACK has no
Restore response payload. Trace keys include epoch, session token and operation ID.

Bootstrap version 6 and channel ABI 10 require rebuilding both client and Manager.
The required companion iceoryx2 request event wakes the Manager after enqueue;
the request queue remains authoritative. Manager maintenance bounds missed-wake
recovery without a fixed 50 us idle poll. Publish retains its source ownership
even if notification fails after enqueue. Its waiter polls the reply eventfd and
Manager pidfd instead of sleeping for 100 us; response delivery and restore
completion use separate notification counters.
Each result mapping has 1024 records and at most 4096 error bytes per record;
mapping admission includes disconnected sessions retained by outstanding work.

Restore batches validate all lease tokens and source geometry before consuming
any lease share. Duplicate tokens are rejected. Preparation failures preserve
valid leases; worker admission and GPU execution are subsequent ownership stages,
so this does not make the whole restore operation retryable after submission.

Run the two-process latency harness in separate terminals:

```bash
cargo run -r -p orbitkv-channel --bin orbitkv-channel-echo -- orbitkv/bench
cargo run -r -p orbitkv-channel --bin orbitkv-channel-bench -- orbitkv/bench 100000
```

The benchmark client sends `Shutdown` after the run so the echo process exits.
