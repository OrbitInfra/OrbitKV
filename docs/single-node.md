# Single-node cache: vLLM and SGLang

Run one independent Cache Manager on the inference host. The engine owns HBM and
scheduling; the Manager owns shared DRAM and optional SSD. Standalone caching
needs neither etcd nor a peer gRPC listener. Check [compatibility](compatibility.md)
and [model qualification](models.md) before enabling a new layout or topology.

## Install and start the common manager

Version 0.1.0 is not yet published. Build the complete wheel from the repository
root; this example selects Python 3.11 and CUDA 13:

```bash
git submodule update --init --recursive third-party/mooncake
./scripts/build-wheel.sh --release
```

CUDA 13 is the default and only release runtime. Select the newly built wheel
in `target/wheels` explicitly and use its absolute path in the installation commands
below. A complete wheel includes the Manager, extension and Mooncake libraries;
`maturin build` alone does not stage all of them. The Manager also needs compatible
PyTorch/CUDA 13 and an NVIDIA R580 or newer driver; see [runtime requirements](releases.md#packages).
It does not load
model weights. Install the same OrbitKV build in client and Manager environments.

After creating an engine environment below, start one Manager from that environment:

```bash
orbitkv-cache-manager \
  --addr 127.0.0.1:50055 \
  --http-addr 127.0.0.1:9091 \
  --pool-size 8gb
```

Use the environment's absolute `bin/orbitkv-cache-manager` path if it is not
activated. The default bootstrap socket is `/tmp/orbitkv-50055.sock`.
`--pool-size` controls external pinned memory, not engine HBM.

To add SSD, use a writable path and an explicit capacity:

```bash
orbitkv-cache-manager \
  --addr 127.0.0.1:50055 --http-addr 127.0.0.1:9091 --pool-size 8gb \
  --ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 100gb
```

The default SSD backend tries cuFile and uses io_uring when unavailable. cuFile is
an optional Manager library; it needs no separate service. Native GDS has separate
hardware requirements and remains unqualified. The SSD file is truncated on Manager
startup; it is not durable across Manager restarts. See [GPU storage](gds.md).

## vLLM

Create an environment with the selected official release and your complete wheel:

```bash
uv venv .venv/vllm-release --python 3.11
uv pip install --python .venv/vllm-release/bin/python 'vllm==0.31.0' --torch-backend=cu130
uv pip install --python .venv/vllm-release/bin/python /absolute/path/to/the-built-wheel.whl
```

Start the Manager with `.venv/vllm-release/bin/orbitkv-cache-manager`.
In another terminal, start the dense V1 eager profile:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 .venv/vllm-release/bin/vllm serve /path/to/immutable-model \
  --enforce-eager --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "OrbitKVConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "orbitkv.vllm"
  }'
```

The supported starting profile has one attention cache group. V2, multiple groups
and recurrent external recovery are rejected. vLLM uses its native HBM prefix
cache first; OrbitKV restores missing compatible blocks and saves newly computed
full blocks. For another Manager port, set `orbitkv.port` in
`kv_connector_extra_config` or `ORBITKV_PORT`; use `orbitkv.bootstrap_socket` for
an explicit socket. See [adapter configuration](adapters.md).

## SGLang

Use Python 3.11+ for SGLang 0.5.22. Start with the installed Qwen3-8B dense
DRAM/io_uring profile in [compatibility](compatibility.md). Broader historical
0.5.21 profiles retain their recorded versions.

Use a separate official release environment:

```bash
uv venv .venv/sglang-release --python 3.11
uv pip install --python .venv/sglang-release/bin/python 'sglang==0.5.22' --torch-backend=cu130
uv pip install --python .venv/sglang-release/bin/python /absolute/path/to/the-built-wheel.whl
```

If no Manager is running, start one from this environment. Enable the linker with
an explicit eager starting profile:

```bash
ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  .venv/sglang-release/bin/sglang serve \
  --model-path /path/to/immutable-model --page-size 64 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --enable-unified-cache-external-linker --radix-cache-backend orbitkv
```

Both linker flags are required. SGLang owns the radix tree and GPU slots; OrbitKV
restores page-aligned state into those slots. Pending reads wait within their
budget, then recompute on expiry. Full hybrid recovery requires separate model
and lifecycle gates; native model output alone does not qualify external recovery.

Both adapters bind model artifacts, computation and layout to cache identity.
Use immutable local artifacts or a full Hub revision. An explicitly verified
`ORBITKV_MODEL_FINGERPRINT` can avoid repeated artifact reads; update it when its
covered artifacts change. Static local LoRA has scoped acceptance; dynamic adapter
mutation and live weight updates require stopping and restarting the deployment.
See [state identity](state-identity.md).

## Check behavior and capacity

```bash
curl --fail http://127.0.0.1:9091/health
curl --fail http://127.0.0.1:9091/metrics
```

Keep the Manager alive, restart the engine, then repeat a multi-block prompt.
Check that `orbitkv_save_bytes_total` rose before the restart and
`orbitkv_load_bytes_total` rose on recovery. Compare output with a cold-control
namespace. A same-process repeated prompt may hit only the engine's HBM.

An accepted host save guarantees its submitted GPU reads completed, not that
best-effort asynchronous SSD publication created a replica. Pages remain held
through the selected transfer's completion. On a miss, the engine computes normally.
Size DRAM, SSD and query budgets for your workload; [metrics](metrics.md) and
[performance methodology](performance.md) describe how to measure them.
Leave optional warming, retention and cost-selection experiments disabled for
this first deployment.

## Deployment variants

The Manager and engine must share a host, UID, socket, iceoryx2 resources and GPU
access. Native clients require process visibility for lifecycle checks. Distinct
CUDA ordinals are allowed when they resolve the same full GPU UUID. Separate PID
namespaces and container deployment need additional qualification; see
[deployment requirements](deployment.md#containers-and-kubernetes).

For multiple engines, optional SSD and peers, use [deployment recipes](deployment.md).
Real-time P/D uses official NIXL/MultiConnector or SGLang disaggregation;
Manager peer-cache bytes use TENT. See [P/D composition](pd.md) before combining them.
