# OrbitKV Cache Manager

This crate runs the cache engine and exposes it to inference processes. The
default single-node path uses `endpoint/` for iceoryx2 commands and UDS lifecycle
frames. `endpoint/pending.rs` owns in-flight query
state; `cache/lifecycle.rs` serializes
registration and cleanup. `wire.rs` converts shared protobuf registration
messages into cache-layer inputs. The distributed listener serves only peer
transfer control RPCs. OrbiTE transfers KV bytes between nodes.

Normal shutdown closes the process endpoint's Publish admission gate before it
drains accepted continuations. Admission and close share one atomic state, so a
request that passed an earlier control-loop check either registers before close
and is drained, or is rejected without consuming its descriptor. Lifecycle
connections use a durable close latch: partial header/payload frames are dropped,
while a complete request finishes dispatch and its bounded response before the
idle connection closes and releases its session owners.

## Building

The binary embeds CPython via PyO3 so it can reconstruct registered CUDA tensors with Torch. Before running cargo commands, point PyO3 to the exact interpreter you want (usually the repo's `.venv`) so linking works and the runtime can import `orbitkv.client.gpu`:

```bash
# Explicitly set your Python interpreter path if needed:
export PYO3_PYTHON="$(pwd)/.venv/bin/python"

export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

cargo run -r --bin orbitkv-cache-manager -- --pool-size 30gb
```

Adjust the Python path if your venv uses a different minor version.

For peer control, configure `--etcd-endpoints`, `--node-id`, and a routable `--addr`. Background etcd publication and snapshot/Watch maintain a complete local global index; the peer port serves source grants and release.
`--devices` selects CUDA device IDs; omitting it detects available devices
automatically. Keep the Python extension and Cache Manager from
the same build because the bootstrap protocol is versioned.
