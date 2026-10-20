# Compatibility and support

Use this page to choose an engine, model and deployment profile. **Implemented**
means code exists; **qualified** means independent review accepted the stated
frozen workload. Acceptance does not extend to other models, topologies or faults.
The [completion plan](completion-plan.md) owns remaining delivery work.

## Official engine releases

| Engine | Pinned release | External-cache entry point | Qualified starting profile |
| --- | --- | --- | --- |
| vLLM | `0.31.0` | `OrbitKVConnector` through the official KV Connector API | V1 runner, one attention cache group, dense TP=1/PP=1 |
| SGLang | `0.5.22` | `OrbitKVLinker` through UnifiedRadixCache's external linker and public load completion | Qwen3-8B dense TP=1/PP=1, installed DRAM/io_uring bytes and model/restart controls |

Set `VLLM_USE_V2_MODEL_RUNNER=0` before starting vLLM. V2, multiple cache groups
and recurrent external-cache profiles are rejected until released lifecycle
contracts can protect their pages. SGLang 0.5.22 requires Python 3.11+ and
supports only Full/SWA external-linker components; recurrent state is rejected
before connecting to a Manager. SGLang's public lifecycle migration remains
open: the current adapter still consumes bounded private graph-initialization
and pending-query admission hooks. See [adapter contracts](adapters.md) and the
[interface audit](engine-release-audit.md).

These are OrbitKV's selected release pins, not a promise that upstream has no
newer release. Historical results retain their original engine versions.

## Deployment profiles

SGLang 0.5.22 repeats the Qwen3-8B single-engine normal path: complete GPU byte
checks, independent consumer leases, known-unsubmitted cancellation, HBM flush,
engine restart, four concurrent recovery requests and a changed-identity miss.
The installed official engine files remain unchanged. Native worker exits
`-9` and CUDA IPC shutdown warnings remain recorded lifetime limits; this does
not qualify physical cancellation or page reuse. Broader SGLang profiles below
retain their measured 0.5.21 versions until their own gates are repeated.

| Profile | Current evidence and limit |
| --- | --- |
| Single engine, DRAM or io_uring SSD | Qwen3-8B repeated on official SGLang 0.5.22; other accepted models retain their recorded release. Manager stays alive during engine restart |
| Two engines sharing one Manager | H20 900-second dense eager direct/kernel io_uring pressure profile accepted for correctness, progress and bounded resources; fairness and tail isolation remain open |
| CUDA Graph recovery | Explicit serial batch-1 FULL decode Graph recovery accepted for the fixed H20 direct/kernel io_uring profile; arbitrary graph shapes and hybrid replay remain open |
| Two hosts, peer cache | Both official engines have scoped two-H20 RDMA dense TP=1/PP=1 DRAM/io_uring restore, consumer restart and controlled empty-source restart acceptance; this is not cache HA qualification |
| Metadata scaling | Same-host correctness accepted; support remains four owners. Sixteen-owner ordinary visibility passes, but DRAM/SSD isolation fails and sixteen owners remain unqualified |
| TP/PP/EP | Registered loopback query control and a TP=1 worker handshake have scoped acceptance; physical multi-rank and cross-host serving remain unqualified |
| Native P/D plus cache | Official vLLM NIXL/MultiConnector and SGLang native disaggregation are composition candidates; cancellation, peer loss, delayed ACK and page reuse need separate fault gates |
| Separate containers | Shared socket, IPC resources, credentials and process visibility are required; separate-container deployment remains unqualified |
| Native GDS/cuFile | Backend implemented; working io_uring or compatibility-mode cuFile is not evidence of native GDS qualification |

Use [deployment recipes](deployment.md), [shared-cache evidence](shared-cache-qualification.md)
and [P/D boundaries](pd.md) for each profile's exact configuration. Manager shared-cache
traffic uses TENT; real-time P/D uses the engine's native backend.

## Model and identity limits

Dense Qwen3-8B and Qwen3-32B have official-engine external-recovery evidence.
Qwen3.8-27B BF16/FP8 have accepted **native output controls only**; complete
external hybrid recovery and matched performance remain unqualified.
DeepSeek-V4.1-Flash is a separate readiness/recovery target. See [model qualification](models.md)
for immutable revisions and the distinction between input, output and recovery gates.

Immutable local static LoRA identity has a scoped dense eager acceptance. Dynamic
adapter mutation and live weight replacement require stopping and restarting the
deployment; an idle vLLM external reset does not qualify live mutation. Different
engines do not automatically share interchangeable KV bytes.

## Package and runtime

| Runtime | Distribution | Import |
| --- | --- | --- |
| CUDA 12 | `orbitkv-llm` | `orbitkv` |
| CUDA 13 | `orbitkv-llm-cu13` | `orbitkv` |

Install only one distribution per environment. Version 0.1.0 is being prepared
for release; build the complete wheel using the [quickstart](single-node.md).
Upgrade native clients and Manager together. The wheel supplies native binaries;
the host supplies compatible Python, CUDA and selected RDMA libraries, and the
Manager also requires compatible PyTorch. See [release artifact requirements](releases.md).

The SSD file is recreated on Manager startup. Engine-restart reuse is supported
within qualified profiles; Manager restart does not preserve this disk cache.
