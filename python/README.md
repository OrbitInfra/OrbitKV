# OrbitKV

[![Python package v0.1.0 — not yet published](https://img.shields.io/badge/Python_package-v0.1.0%20%28unreleased%29-203b30)](https://feichai0017.github.io/orbitkv/docs/releases/)

**KV cache for vLLM and SGLang, backed by Rust.** Reuse computed prefixes from
pinned DRAM and optional SSD after GPU eviction or an engine restart. Both
adapters use the same native Cache Manager API. Unencoded DRAM Restore runs in
the engine process using shared payload arenas; SSD/codec Restore and Publish
retain Manager workers and CUDA IPC tensor bindings.

[Documentation](https://feichai0017.github.io/orbitkv/docs/) ·
[Quickstart](https://feichai0017.github.io/orbitkv/docs/single-node/) ·
[Architecture](https://feichai0017.github.io/orbitkv/architecture/) ·
[Source](https://github.com/feichai0017/orbitkv)

## Features

- DRAM/SSD prefix reuse with engine-owned GPU allocation.
- Automatic Rust SSD backend selection for both engines, with native cuFile
  writes/restores where available and io_uring otherwise; see the
  [GPU storage configuration and qualification gates](../docs/gds.md).
- Compiled recovery ranges for supported attention, window and checkpoint layouts.
- Optional GPU ANS lossless compression and experimental FP8/3-bit/4-bit
  TurboQuant storage in DRAM, SSD and peer transfers; see [storage formats](../docs/storage-formats.md).
- Native query ownership, source grants, byte budgets and completion fences.
- Engine-local raw Restore with actual tensor ownership and an explicit engine
  readiness stream; both adapters use the whole-operation completion gate.
- Prometheus metrics and optional request timelines.
- Experimental peer cache sharing through Mooncake TENT.
- Native P/D composition candidates use official vLLM NIXL/MultiConnector and
  SGLang disaggregation, with independent cache adapters. See [P/D qualification](../docs/pd.md).

The official vLLM cache profile requires `VLLM_USE_V2_MODEL_RUNNER=0`
and one attention cache group. Set this environment variable before starting
vLLM; unsupported V2/recurrent profiles fail at startup.

Pinned engine releases: **vLLM 0.31.0** and **SGLang 0.5.21**. See the
[upgrade qualification scope](../docs/completion-plan.md#s51--release-and-interface-audit).
The recorded [engine-local Restore path](../docs/engine-local-restore.md) passes
single-H20 Qwen3-8B DRAM correctness and engine-restart reuse with vLLM 0.29.0
and SGLang 0.5.20; those historical results keep their original engine versions.
The [recorded vLLM end-to-end results](../docs/single-node-performance.md#matched-vllm-end-to-end-comparison)
compare native HBM, native CPU offload, OrbitKV and LMCache MP. Multiple-GPU,
huge-page and broader serving qualification remain separate. See
[the measured Restore improvements and remaining overhead](../docs/communication-performance.md),
[model qualification](https://feichai0017.github.io/orbitkv/docs/models/)
and [deployment support](https://feichai0017.github.io/orbitkv/docs/deployment/).
Interfaces may change before 1.0.

## Native registration and Restore contract

`register_context_batch(..., tensors=[...])` takes the real tensor/exporter
objects as well as their IPC metadata. Keep tensor order aligned with layer
registration. `start_restore(..., ready_stream=...)` takes the CUDA stream of
the engine's previous destination-page users; the native worker establishes
readiness before copying. Both bundled adapters provide these arguments.

The native worker owns accepted operations even if a Python handle is dropped
or `wait_restore` times out. Connectors must retain logical destination page IDs
until a terminal result. Local completion means DMA drained and does not wait
for the Manager's source-retirement ACK. Repeated registration of the same
binding is rejected; unregister and close drain accepted operations first.

Build the native client and Manager together: this cutover uses bootstrap 7,
channel ABI 12, cache schema 9, lifecycle 7 and Restore grant schema 5, with no
old-wire decoder. Fragmented raw Restore plans are partitioned into at most
1 MiB parts under one final drain fence, with 32 MiB operation and 64 MiB
session metadata limits. Idle destination streams need no additional GPU event;
busy streams are fenced with a reusable event. Layer overlap and graph replay
dependencies remain future work.

CUDA registration identifies the tensor by its full GPU UUID. The native owner
retains the Manager's resolved CUDA ordinal for Publish/Restore, while the client
keeps its own ordinal for local execution. Different numeric, UUID, reordered or
partial `CUDA_VISIBLE_DEVICES` views do not need matching ordinal numbers.

## Installation

| CUDA runtime | Distribution | Python import |
| --- | --- | --- |
| CUDA 12 | `orbitkv-llm` | `orbitkv` |
| CUDA 13 | `orbitkv-llm-cu13` | `orbitkv` |

Install only one distribution and one engine extra (`vllm` or `sglang`) per
environment. The extras pin the validated engine releases. The wheel includes
the native extension, Cache Manager and Mooncake runtime; the Manager also
needs compatible PyTorch. The base package does not install PyTorch.
The host supplies matching shared Python, CUDA driver/runtime and RDMA libraries
(`libibverbs`, `librdmacm` and the selected NIC provider). Other linked native
dependencies are repaired into the wheel with their redistribution notices;
its platform tag reflects all bundled binaries.

Version 0.1.0 is being prepared for release. Build a complete wheel from the
repository, then install the produced file:

```bash
git submodule update --init --recursive third-party/mooncake
./scripts/build-wheel.sh --release --no-default-features --features cuda-13,mooncake
python -m pip install /absolute/path/to/the-built-wheel.whl
```

Use `./scripts/build-wheel.sh --release` for CUDA 12. See the
[installation guide](https://feichai0017.github.io/orbitkv/docs/single-node/)
for complete engine environments and the
[release guide](https://feichai0017.github.io/orbitkv/docs/releases/) for artifact checks.

## Quickstart

Start an independent Manager in a compatible PyTorch/CUDA environment:

```bash
orbitkv-cache-manager --addr 127.0.0.1:50055 --pool-size 8gb
```

To add SSD capacity, append
`--ssd-cache-path /data/orbitkv/cache.bin --ssd-cache-capacity 100gb`.
The Manager automatically tries native cuFile and falls back to io_uring;
normal deployments do not need `--ssd-backend` or a separate cuFile service.
Engines on the same host can connect to this Manager and share its external
capacity; see [deployment requirements](../docs/deployment.md) for runtime
compatibility, shared resources and current qualification limits.

Then enable the adapter on the same host:

```bash
# vLLM
vllm serve /path/to/immutable-model --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"OrbitKVConnector","kv_role":"kv_both","kv_connector_module_path":"orbitkv.vllm"}'

# SGLang, in its own environment
ORBITKV_SGLANG_ENDPOINT=unix:///tmp/orbitkv-50055.sock \
  sglang serve --model-path /path/to/immutable-model --page-size 64 \
  --enable-unified-cache-external-linker --radix-cache-backend orbitkv
```

Keep the Manager alive, restart the engine and repeat a multi-block prompt to
verify an external restore. SSD contents are recreated when the Manager restarts.
The engine remains responsible for HBM and scheduling. Automatic request
preparation is experimental and disabled by default. A fixed local LoRA set can
use the [static adapter identity profile](../docs/adapters.md#immutable-static-lora);
it requires immutable files and prohibits runtime adapter mutation and sessions.
The completion plan records its installed-model qualification.

## Development

[Adapter reference](https://feichai0017.github.io/orbitkv/docs/adapters/) ·
[Test gates](https://github.com/feichai0017/orbitkv/blob/main/python/tests/README.md) ·
[Benchmarks](https://github.com/feichai0017/orbitkv/tree/main/benches) ·
[Roadmap](https://feichai0017.github.io/orbitkv/docs/roadmap/)

The runtime package contains only the native client and engine adapters. Tests
live in `python/tests/`; workloads and results live in `benches/`.

## License

[Apache-2.0](https://github.com/feichai0017/orbitkv/blob/main/LICENSE).
