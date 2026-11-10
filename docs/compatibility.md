# Compatibility and support

All current OrbiTE **GPU, RDMA, GDS and installed profiles are unqualified**.
CPU/TCP, io_uring and CUDA-shim controls are CI evidence. Historical TENT
acceptances retain their original releases and scope in [performance](performance.md);
none carries over to the current data path.

## Official engine releases

| Engine | Current package/source pin | Implemented boundary |
| --- | --- | --- |
| vLLM | 0.33.0 | KV Connector, V1 runner by default; single attention group or [Full + SWA](adapters.md#released-hma-contract); [V2 opt-in](adapters.md#released-v2-lifetime-contract) admits uniform Full attention only and remains unqualified; recurrent, zeroing and V2 multi-group rejected |
| SGLang | 0.5.23 | UnifiedRadixCache external linker, Full/SWA and static Mamba checkpoints; MLA per-rank; replay, draft, moving pools and combined Full/SWA/Mamba serving rejected; private pre-capture and admission hooks remain |

The pins select adapter source contracts; installed GPU conformance remains
queued in [M1](completion-plan.md#m1--engine-baseline). Follow
[adapter contracts](adapters.md) and [model recipes](models.md), not upstream
model support alone, when selecting a layout.

## Deployment boundaries

| Profile | Implementation and limit |
| --- | --- |
| Local DRAM / SSD | Implemented; default SSD is recreated at startup. Opt-in persistent SSD has CPU crash controls; installed NVMe restart serving is unqualified |
| Shared Manager | Local engines can share capacity; fairness and tail isolation remain open |
| Peer cache | Inventory, grants and OrbiTE TCP/RDMA; current hardware recovery and cache HA unqualified |
| Metadata scaling | Four-owner scope; historical isolation FAIL and sixteen-owner UNQUALIFIED remain unchanged |
| Graph / TP / PP / EP | Some layout, readiness and rank-control paths exist; each installed shape/rank profile needs its own evidence |
| Native P/D plus cache | vLLM NIXL/MultiConnector or SGLang native disaggregation; cancellation, peer loss, delayed ACK and page reuse unqualified |
| Containers | Shared socket, IPC resources, credentials and process visibility required; separate-container use unqualified |
| GDS | Implemented opt-in; io_uring or compatibility-mode bytes do not prove native GDS |

Use [deployment](deployment.md), [fault gates](fault-qualification.md) and
[P/D composition](pd.md). Manager sharing uses OrbiTE; live P/D uses the engine's
native transport. Immutable local static LoRA participates in cache identity;
dynamic adapter mutation and live weight replacement remain unsupported.
Different engines do not automatically share compatible KV bytes.

## Package and runtime

| Runtime | Distribution | Import |
| --- | --- | --- |
| CUDA 13 | `orbitkv` | `orbitkv` |

CUDA 13 is the only release runtime; normally use NVIDIA R580 or newer with
toolkit-compatible PTX/NVRTC support. Supported older data-center drivers need
the explicit [forward-compatibility runtime profile](releases.md#packages).
Version 0.1.0 is being prepared for release; build the complete wheel using the [quickstart](single-node.md).
Upgrade native clients and Manager together. The wheel supplies native binaries;
the host supplies compatible Python, CUDA and selected RDMA libraries, and the
Manager also requires compatible PyTorch. See [release artifact requirements](releases.md).

By default, SSD files are recreated on Manager startup. `--ssd-persistent` enables
[durable commits and verified recovery](storage-persistence.md). CPU process-crash
controls do not qualify real NVMe serving or power-loss durability.
