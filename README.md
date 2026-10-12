<p align="center">
  <img src="website/public/readme-banner.svg" alt="OrbitKV — Compile the lifetime of state" width="100%" />
</p>

<p align="center">
  <a href="https://feichai0017.github.io/orbitkv/">Website</a> ·
  <a href="https://feichai0017.github.io/orbitkv/docs/">Documentation</a> ·
  <a href="docs/single-node.md">Quickstart</a> ·
  <a href="docs/compatibility.md">Compatibility</a> ·
  <a href="docs/contributing.md">Contributing</a>
</p>

<p align="center">
  <a href="https://github.com/feichai0017/orbitkv/actions/workflows/ci.yml"><img src="https://github.com/feichai0017/orbitkv/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="docs/releases.md"><img src="https://img.shields.io/badge/Python_package-v0.1.0%20%28unreleased%29-203b30" alt="Python package v0.1.0 — not yet published" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-203b30" alt="Apache 2.0 license" /></a>
</p>

## About

**OrbitKV extends the KV cache of vLLM and SGLang beyond GPU memory.** Store
reusable prefixes in DRAM and optional SSD, then restore them after GPU eviction
or an engine restart. This serves repeated documents, shared system prompts and
long conversation prefixes.

Run an independent Cache Manager on each inference host. Engines own GPU pages
and scheduling; Rust owns external replicas, byte budgets and transfer completion.
The pinned official engines are **vLLM 0.31.0** and **SGLang 0.5.21**. Start with
dense TP=1/PP=1 inference and check the [support matrix](docs/compatibility.md).
Version 0.1.0 is not yet published; interfaces may change before 1.0.

## Features

- **DRAM and SSD reuse:** keep the Manager alive across engine restarts.
- **Native transfers:** engine-local raw DRAM Restore; Manager-owned Publish,
  SSD and codec work, with ownership retained through completion.
- **Bounded resources:** shared cache capacity and per-instance query budgets.
- **Peer cache sharing:** local inventory indexes discover replicas; Mooncake
  TENT transfers bytes through TCP or RDMA in qualified profiles.

Optional [storage codecs](docs/storage-formats.md), [reuse policies](docs/cache-policies.md)
and [CUDA copy kernels](docs/kernel-optimization.md) have separate qualification
limits. Native P/D uses the engines' own transports and independent cache adapters;
see [P/D composition](docs/pd.md).

## Get started

Follow the [installation and quickstart](docs/single-node.md) to build a complete
wheel for your Python/CUDA runtime. Use a separate environment for each engine
and install the same OrbitKV build in the Manager environment. The Manager needs
compatible PyTorch and does not load model weights.

Start a Manager:

```bash
orbitkv-cache-manager --addr 127.0.0.1:50055 --http-addr 127.0.0.1:9091 --pool-size 8gb
```

Start vLLM in its official release environment:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve /path/to/immutable-model \
  --enforce-eager --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OrbitKVConnector","kv_role":"kv_both","kv_connector_module_path":"orbitkv.vllm"}'
```

For SGLang, follow its [linker recipe](docs/single-node.md#sglang). To verify an
external hit, keep the Manager alive, restart the engine and repeat a multi-block
prompt; check `orbitkv_load_bytes_total`. A repeated prompt in the same engine can
hit its native GPU cache instead.

## Learn more

| Task | Guide |
| --- | --- |
| Select a release, model and topology | [Compatibility](docs/compatibility.md), [models](docs/models.md) |
| Share capacity or add SSD/peers | [Deployment](docs/deployment.md), [Manager configuration](docs/server.md) |
| Understand ownership and data paths | [Architecture](docs/architecture.md), [transport](docs/transport.md) |
| Measure latency and throughput | [Performance](docs/performance.md), [benchmark methodology](docs/benchmark-evidence.md) |
| Develop and review changes | [Contributing](docs/contributing.md), [completion plan](docs/completion-plan.md) |

Distributed HA, native GDS, broader parallelism and hybrid external recovery
remain subject to separate gates. Current metadata support stays at four owners;
the 16-owner visibility gate passed, but DRAM/SSD isolation failed. Component
speedups do not establish a win over native caching or LMCache in serving.

## License

[Apache 2.0](LICENSE). Bundled dependencies retain their own licenses and notices.
