<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo-dark.svg" />
    <img src="docs/_static/logo-light.svg" alt="OrbitKV" width="320" />
  </picture>
</p>

<p align="center"><strong>A distributed KV cache for vLLM and SGLang</strong></p>

<p align="center">
  <a href="https://orbitinfra.github.io/OrbitKV/">Documentation</a> |
  <a href="docs/single-node.md">Quickstart</a> |
  <a href="docs/completion-plan.md">Plan</a> |
  <a href="docs/contributing.md">Contributing</a>
</p>

<p align="center">
  <a href="https://github.com/OrbitInfra/OrbitKV/actions/workflows/ci.yml"><img src="https://github.com/OrbitInfra/OrbitKV/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="docs/releases.md"><img src="https://img.shields.io/badge/Python_package-v0.1.0%20%28unreleased%29-203b30" alt="Python package v0.1.0 — not yet published" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-203b30" alt="Apache 2.0 license" /></a>
</p>

## Updates

- **2026-11-07:** [OrbiTE becomes the sole transfer engine](https://github.com/OrbitInfra/OrbitKV/pull/326); hardware re-qualification remains open.

## About

**OrbitKV extends the KV cache of vLLM and SGLang beyond GPU memory.** It is for
serving workloads that reuse documents, system prompts and conversation prefixes.
An independent Cache Manager keeps reusable state in DRAM and optional SSD
across engine restarts; inference engines keep scheduling and GPU allocation.

Rust owns cache identity, placement, leases and bounded admission.
**OrbiTE is the only byte-transfer engine** for local GPU↔DRAM copies, SSD and
peer reads. Current GPU, RDMA, GDS and installed profiles are **unqualified**;
CPU/TCP and io_uring controls are CI evidence. Historical TENT results do not
qualify OrbiTE. See [support status](docs/compatibility.md) and [evidence](docs/performance.md).

## Key features

- **Reuse beyond GPU memory:** retain compatible prefixes in DRAM and optional
  SSD while the Manager stays alive. SSD is not durable across Manager restarts.
- **Share cache capacity:** attach local engines to a Manager or discover peer
  replicas through bounded inventory indexes and authoritative source grants.
- **Keep ownership explicit:** hold sources, destinations and registrations until
  physical completion; a timeout is never permission to reuse their bytes.
- **Describe recovery requirements:** compile supported prefix, window and
  checkpoint layouts into complete recovery boundaries. Engine guards still apply.

[Features and limits](docs/features.md) cover optional codecs and retention
policies. Live prefill/decode uses the [inference engine's native transport](docs/pd.md).

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/architecture-dark.svg" />
  <img src="docs/_static/architecture-light.svg" alt="OrbitKV owns cache semantics; OrbiTE moves bytes between GPU, DRAM, SSD and peers." width="960" />
</picture>

Implemented ownership boundaries; hardware paths remain **unqualified**.
See [architecture](docs/architecture.md) and [transfer contracts](docs/transport.md).

## Getting started

[OrbitKV 0.1.0 is unreleased](docs/releases.md). On Linux with a compatible
[CUDA 13 build/runtime environment](docs/releases.md#packages), build a complete
wheel from the repository root and install it into your chosen engine environment:

```bash
PYO3_PYTHON=python3.11 ./scripts/build-wheel.sh --release
uv pip install --python /path/to/engine-env/bin/python /absolute/path/to/the-built-wheel.whl
```

Use separate environments for the pinned engines and the same wheel for client
and Manager. The [quickstart](docs/single-node.md) supplies prerequisites, engine
installation and hit checks. With the chosen environment activated, start a Manager:

```bash
orbitkv-cache-manager --addr 127.0.0.1:50055 --http-addr 127.0.0.1:9091 --pool-size 8gb
```

Then choose one engine in another terminal with the same environment activated.
These are **unqualified hardware recipes**, not benchmark results.

**vLLM** ([pinned release and limits](docs/compatibility.md#official-engine-releases)):

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve /path/to/immutable-model \
  --enforce-eager --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OrbitKVConnector","kv_role":"kv_both","kv_connector_module_path":"orbitkv.vllm"}'
```

**SGLang** ([pinned release and limits](docs/compatibility.md#official-engine-releases)):

```bash
ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  sglang serve --model-path /path/to/immutable-model --page-size 64 \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --enable-unified-cache-external-linker --radix-cache-backend orbitkv
```

[Verify an external hit](docs/single-node.md#check-behavior-and-capacity): save a
multi-block prompt, keep the Manager alive, restart the engine and repeat it.
Check that `orbitkv_load_bytes_total` increases; a same-process repeat can hit
only native GPU cache. [Manager options](docs/server.md) explain the example budgets.

The [documentation](https://orbitinfra.github.io/OrbitKV/) covers engine integration,
storage, distributed cache, deployment, observability and internals. The
[support matrix](docs/compatibility.md) selects **vLLM 0.33.0** (V1, one attention
group) and **SGLang 0.5.23** (Full/SWA external linker); recurrent external
recovery remains rejected. See [model recipes and limits](docs/models.md).

## Contributing

Follow the [contributor guide](docs/contributing.md) and the
[completion plan](docs/completion-plan.md). For the data flow, read
[transport](docs/transport.md) → [engine-local restore](docs/engine-local-restore.md)
→ [distributed cache](docs/distributed-cache.md) → [adapter contracts](docs/adapters.md).

## License

[Apache 2.0](LICENSE). Bundled dependencies retain their own licenses and notices.
