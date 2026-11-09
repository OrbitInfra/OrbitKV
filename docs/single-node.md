# Single-node cache: vLLM and SGLang

Run one independent Cache Manager on the inference host. The engine owns HBM and
scheduling; the Manager owns shared DRAM and optional SSD. Standalone caching
needs neither etcd nor a peer gRPC listener. Check [compatibility](compatibility.md)
and [model qualification](models.md) before enabling a new layout or topology.
These are current configuration recipes; OrbiTE GPU/installed paths require
hardware re-qualification before any support or performance claim.

## Prerequisites

- A Linux host with an NVIDIA GPU, a compatible CUDA 13 toolkit/driver and enough
  GPU memory for the model. Follow the [build/runtime requirements](releases.md#packages);
  install Rust, uv 0.13.7 or newer, and the native build prerequisites listed there before building.
- Python 3.11 with shared `libpython`, used for both engine environments below.
- An immutable local dense model compatible with the [selected release](compatibility.md#official-engine-releases),
  such as the [Qwen3-8B recipe](models.md). These profiles remain **unqualified on OrbiTE**.
- Client and Manager on the same host and UID, with shared socket/process visibility.
  The example Manager uses `--pool-size 8gb`; adjust it to your host-memory budget.

Use the repository root as your working directory. Replace the model path and,
after building, the wheel path below. Set these in each terminal used by the recipe:

```bash
export ORBITKV_MODEL=/absolute/path/to/immutable-model
export ORBITKV_WHEEL=/absolute/path/to/the-built-wheel.whl
```

## Install and start the common manager

Version 0.1.0 is not yet published. Build the complete wheel from the repository
root; this example selects Python 3.11 and CUDA 13:

```bash
PYO3_PYTHON=python3.11 ./scripts/build-wheel.sh --release
```

CUDA 13 is the default and only release runtime. Select the newly built wheel
in `target/wheels` explicitly and use its absolute path in the installation commands
below. A complete wheel includes the Manager and extension with OrbiTE linked in;
`maturin build` alone does not stage all of them. The Manager also needs compatible
PyTorch/CUDA 13 and a compatible driver; see [runtime requirements](releases.md#packages).
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

The default SSD backend uses engine-owned io_uring and CUDA staging. Explicit
`--ssd-backend cufile` selects native-only GDS in the engine; it needs no separate service. Native GDS has separate
hardware requirements and remains unqualified. The SSD file is truncated on Manager
startup; it is not durable across Manager restarts. See [GPU storage](gds.md).

## vLLM

**Prerequisites:** complete the common setup above; this recipe uses the pinned
vLLM 0.33.0 V1 release with one attention group. Use a separate terminal for the
Manager and keep it running during engine restarts.

### Install and run vLLM

```bash
uv venv .venv/vllm-release --python 3.11
uv pip install --python .venv/vllm-release/bin/python 'vllm==0.33.0' --torch-backend=cu130
uv pip install --python .venv/vllm-release/bin/python "$ORBITKV_WHEEL"
```

Start the Manager in its own terminal:

```bash
.venv/vllm-release/bin/orbitkv-cache-manager \
  --addr 127.0.0.1:50055 --http-addr 127.0.0.1:9091 --pool-size 8gb
```

In another terminal with `ORBITKV_MODEL` set, start the dense V1 eager profile:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 .venv/vllm-release/bin/vllm serve "$ORBITKV_MODEL" \
  --host 127.0.0.1 --port 8000 \
  --enforce-eager --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "OrbitKVConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "orbitkv.vllm"
  }'
```

The starting configuration has one attention cache group. V2, multiple groups
and recurrent external recovery are rejected. vLLM uses its native HBM prefix
cache first; OrbitKV restores missing compatible blocks and saves newly computed
full blocks. For another Manager port, set `orbitkv.port` in
`kv_connector_extra_config` or `ORBITKV_PORT`; use `orbitkv.bootstrap_socket` for
an explicit socket. See [adapter configuration](adapters.md).

### Verify a vLLM hit

Wait for the engine to be ready, then send a deterministic multi-block prompt:

```bash
python3 - <<'PY' | curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' -d @-
import json
import os
print(json.dumps({"model": os.environ["ORBITKV_MODEL"],
                  "prompt": "A cache retains a repeated document. " * 128 + "Summarize it.",
                  "max_tokens": 8, "temperature": 0}))
PY
```

Follow the [save → engine restart → repeat check](#check-behavior-and-capacity)
below using this exact request. **Next:** read [adapter contracts](adapters.md)
before enabling other request modes, or add SSD with the Manager recipe above.

## SGLang

**Prerequisites:** complete the common setup above and use Python 3.11+ for
SGLang 0.5.23. Start with the dense DRAM/io_uring configuration in
[compatibility](compatibility.md); the released external linker supports Full/SWA,
and recurrent startup is rejected. Historical results do not qualify this recipe.

### Install and run SGLang

Use a separate environment, with no other engine occupying the GPU:

```bash
uv venv .venv/sglang-release --python 3.11
uv pip install --python .venv/sglang-release/bin/python 'sglang==0.5.23' --torch-backend=cu130
uv pip install --python .venv/sglang-release/bin/python "$ORBITKV_WHEEL"
```

If no Manager is running, start one from this environment in its own terminal:

```bash
.venv/sglang-release/bin/orbitkv-cache-manager \
  --addr 127.0.0.1:50055 --http-addr 127.0.0.1:9091 --pool-size 8gb
```

In another terminal with `ORBITKV_MODEL` set, enable the eager linker profile:

```bash
ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  .venv/sglang-release/bin/sglang serve \
  --model-path "$ORBITKV_MODEL" --page-size 64 --host 127.0.0.1 --port 30000 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --enable-unified-cache-external-linker --radix-cache-backend orbitkv
```

Both linker flags are required. SGLang owns the radix tree and GPU slots; OrbitKV
restores page-aligned state into those slots. Pending reads wait within their
budget, then recompute on expiry. Full hybrid recovery requires separate model
and lifecycle gates; native model output alone does not qualify external recovery.

### Verify a SGLang hit

Wait for the engine to be ready, then send the same deterministic prompt:

```bash
python3 - <<'PY' | curl --fail http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' -d @-
import json
print(json.dumps({"text": "A cache retains a repeated document. " * 128 + "Summarize it.",
                  "sampling_params": {"max_new_tokens": 8, "temperature": 0}}))
PY
```

Use the [save → engine restart → repeat check](#check-behavior-and-capacity)
below. **Next:** consult [model limits](models.md) and [adapter contracts](adapters.md)
before trying other layouts; configure capacity through the [Manager](server.md).

Both adapters bind model artifacts, computation and layout to cache identity.
Use immutable local artifacts or a full Hub revision. An explicitly verified
`ORBITKV_MODEL_FINGERPRINT` can avoid repeated artifact reads; update it when its
covered artifacts change. Static local LoRA has scoped acceptance; dynamic adapter
mutation and live weight updates require stopping and restarting the deployment.
See [state identity](state-identity.md).

## Check behavior and capacity

Use a fresh Manager and no other traffic for this control. Check health and note
the starting [save/load counters](metrics.md#available-metrics):

```bash
curl --fail http://127.0.0.1:9091/health
curl --fail http://127.0.0.1:9091/metrics | grep -E '^orbitkv_(save|load)_bytes_total'
```

1. Send the request from your engine recipe. Wait for `orbitkv_save_bytes_total`
   to increase, then wait for submitted saves with the command below.
2. Stop only the engine normally, keep the Manager running, and restart the
   exact same engine command with the same model, wheel and settings.
3. Repeat the exact request. Require `orbitkv_load_bytes_total` to increase and
   compare the greedy output with the first, cold request. Otherwise this check
   has not established external reuse; inspect [metrics](metrics.md) and logs.

```bash
curl --fail -X POST http://127.0.0.1:9091/cache/sync
```

A repeated request without engine restart may hit only native HBM. These steps
check one normal path; they do not qualify faults, arbitrary models or hardware.

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
Manager peer-cache bytes use OrbiTE. See [P/D composition](pd.md) before combining them.
