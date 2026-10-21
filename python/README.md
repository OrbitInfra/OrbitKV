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
| CUDA 13 | `orbitkv-llm` | `orbitkv` |

CUDA 13 is the only supported runtime, with NVIDIA R580 or newer and a driver
compatible with the selected toolkit's runtime-compiled kernels. Version 0.1.0
is being prepared for release; build a complete wheel from the repository root:

```bash
git submodule update --init --recursive third-party/mooncake
./scripts/build-wheel.sh --release
python -m pip install /absolute/path/to/the-built-wheel.whl
```

The wheel includes the
extension, Manager and Mooncake libraries; the host supplies compatible Python,
CUDA and RDMA libraries. The Manager also requires compatible PyTorch, which the
base package does not install. See [release requirements](../docs/releases.md).

## Engine integration

Use separate official environments for **vLLM 0.31.0** and **SGLang 0.5.22**,
with the same OrbitKV build in the Manager environment. vLLM requires
`VLLM_USE_V2_MODEL_RUNNER=0` and one attention cache group. SGLang uses the
UnifiedRadixCache external linker and its public load-completion callback.
SGLang requires Python 3.11+. Its 0.5.22 Qwen3-8B normal-path acceptance is
scoped in [compatibility](../docs/compatibility.md); broader 0.5.21 measurements
remain historical. Pre-capture and pending-admission hooks remain until released
replacements are available.
Follow the [engine quickstart](../docs/single-node.md) for complete commands and
an external-hit check. Native P/D owns its transport; OrbitKV owns independent
cache adapters and Manager peer-cache traffic through TENT.

## Native registration and Restore contract

`register_context_batch(..., tensors=[...])` returns an opaque `GpuRegistration`
that retains actual tensor/exporter objects in registration order.
Explicit registration exports storage IPC metadata in the native
client; Python supplies retained tensors and layouts, without serialized pickle
wrappers. Matching lifecycle-9 packages are required. Storage and view offsets,
GPU UUID, dtype/layout, counter and readiness fields are validated before import.
Scoped installed GPU/adapter and export-error ownership gates are independently
accepted; see S3 in the [completion plan](../docs/completion-plan.md). Physical
fault termination and allocation/page-generation qualification remain open.

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
