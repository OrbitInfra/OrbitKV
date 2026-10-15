# OrbitKV Python package

OrbitKV connects official vLLM and SGLang to an independent Rust Cache Manager.
Reuse prefixes from DRAM and optional SSD after GPU eviction or engine restart.
Engines own GPU memory and scheduling; Rust owns cache state, batches, budgets
and transfer completion.

[Quickstart](../docs/single-node.md) · [Compatibility](../docs/compatibility.md) ·
[Adapter contracts](../docs/adapters.md) · [Test gates](tests/README.md)

## Installation

| CUDA runtime | Distribution | Import |
| --- | --- | --- |
| CUDA 12 | `orbitkv-llm` | `orbitkv` |
| CUDA 13 | `orbitkv-llm-cu13` | `orbitkv` |

Install one distribution per environment. Version 0.1.0 is being prepared for
release; build a complete wheel from the repository root:

```bash
git submodule update --init --recursive third-party/mooncake
./scripts/build-wheel.sh --release --no-default-features --features cuda-13,mooncake
python -m pip install /absolute/path/to/the-built-wheel.whl
```

Use `./scripts/build-wheel.sh --release` for CUDA 12. The wheel includes the
extension, Manager and Mooncake libraries; the host supplies compatible Python,
CUDA and RDMA libraries. The Manager also requires compatible PyTorch, which the
base package does not install. See [release requirements](../docs/releases.md).

## Engine integration

Use separate official environments for **vLLM 0.31.0** and **SGLang 0.5.21**,
with the same OrbitKV build in the Manager environment. vLLM requires
`VLLM_USE_V2_MODEL_RUNNER=0` and one attention cache group. SGLang uses the
UnifiedRadixCache external linker; public lifecycle migration remains open.
Follow the [engine quickstart](../docs/single-node.md) for complete commands and
an external-hit check. Native P/D owns its transport; OrbitKV owns independent
cache adapters and Manager peer-cache traffic through TENT.

## Native registration and Restore contract

`register_context_batch(..., tensors=[...])` returns an opaque `GpuRegistration`
that retains actual tensor/exporter objects in registration order.
Use `save(registration, saves)` for Publish. Queued engine callbacks capture that
registration when their producer hands off work; an inactive registration or a
different native client is rejected before Publish. Keeping the object retains
its exporters after unregister, but does not authorize old work or prove device
completion. `start_restore(..., ready_stream=...)` supplies
readiness for previous destination users. Both bundled adapters supply these
arguments. Raw DRAM Restore executes in the engine through imported payload
arenas; Publish and SSD/codec work execute in Manager workers.

Dropping a Python handle or timing out a wait does not release accepted native
operations. Keep logical destination pages held until terminal completion.
Unregister and close drain accepted work. GPU identity uses full UUIDs, so
client and Manager ordinals may differ. Native clients and Manager must be
upgraded together; old wire formats are rejected.

See [ownership and layer/graph readiness](../docs/engine-local-restore.md) and
[the binding stubs](orbitkv/orbitkv.pyi) before implementing another adapter.
