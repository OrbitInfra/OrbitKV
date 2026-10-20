# Model recipes

Select the model architecture as well as the engine release. OrbitKV restores
cached state into engine-owned pages; every state family the model reads must
have a supported layout and recovery boundary. A model loading successfully in
vLLM or SGLang does not establish external-cache recovery.

## Choose a model

| Model/profile | vLLM 0.31.0 | SGLang 0.5.21 | How to use it |
| --- | --- | --- | --- |
| Qwen3-8B, dense attention | Scoped DRAM/io_uring recovery and engine restart accepted | Scoped DRAM/io_uring recovery and engine restart accepted | Start with the dense recipe below |
| Qwen3-32B, BF16 dense attention | H20 TP=1/PP=1 ordinary recovery accepted | H20 TP=1/PP=1 ordinary recovery accepted | Size HBM and host cache for the workload; see measured limits below |
| Other dense checkpoints | Not yet qualified | Not yet qualified | Validate their actual cache layout, identity and output before deployment |
| Qwen3.8-27B, BF16/FP8 weights | Native output controls only; external hybrid profile rejected | Native output controls only; full external recovery unqualified | Hybrid recipe requirements below |
| DeepSeek-V4.1-Flash | Metadata/configuration checked; recovery unqualified | Metadata/configuration checked; recovery unqualified | Requires complete compressed and auxiliary-state contracts |
| Multimodal, dynamic LoRA or live weight updates | Unqualified | Unqualified | Do not reuse text-only or static-adapter qualification |

The selected checkpoint revisions are:

| Checkpoint | Revision |
| --- | --- |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen3-32B | `9216db5781bf21249d130ec9da846c4624c16137` |
| Qwen3.8-27B BF16 | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Qwen3.8-27B FP8 | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| DeepSeek-V4.1-Flash | `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` |

See [compatibility](compatibility.md) for exact execution, topology and fault
limits. These are measured checkpoints, not a claim that every family member
or engine version is supported.

## Dense attention: Qwen3-8B and Qwen3-32B

1. Keep a local immutable model directory containing weights, configuration,
   tokenizer and chat template. Use the same artifacts when restarting.
2. Install OrbitKV and the selected official engine in its own environment;
   start the independent Manager from the [quickstart](single-node.md).
3. For [vLLM](single-node.md#vllm), set `VLLM_USE_V2_MODEL_RUNNER=0` and enable
   `OrbitKVConnector`. Start with eager TP=1/PP=1 and one attention cache group.
4. For [SGLang](single-node.md#sglang), enable the OrbitKV plugin and external
   linker. Start with ordinary contiguous GPU pools and dense TP=1/PP=1.
5. Send a repeated multi-block prompt, keep the Manager alive, restart the
   engine and repeat the request. Verify nonzero `orbitkv_load_bytes_total`
   and output against a cold/native control. An in-process HBM hit alone does
   not show that OrbitKV supplied the cache.

Add [SSD](deployment.md#configure-capacity-then-connect-engines) or
[peer caching](p2p.md) after the local recipe works. The SSD file is recreated
on Manager startup: engine restart reuse requires keeping the Manager alive.
Keep model/computation/layout identities equal; different engine KV layouts are
not automatically interchangeable.

Qwen3-32B's ordinary H20 profile uses BF16/raw, TP=1/PP=1. Its original vLLM
control disables Graph; SGLang retains its recorded graph setting. The separate
explicit FULL decode Graph acceptance covers only the fixed batch-1/eight-token
profile. Use [performance](performance.md) for the reviewed 4K/8K comparisons;
large weights, more GPUs or a new graph shape do not extend those results.

## Hybrid attention and recurrent state

Hybrid recovery must restore attention pages, legal window/sink ranges and
recurrent checkpoints together. Missing state cannot be replaced by a prefix
hit count. Qwen3.8-27B has attention and GDN state; its accepted BF16/FP8 native
controls use BF16 KV and do not establish external hybrid recovery or quality
across precisions/engines.

vLLM's current adapter rejects V2, multiple cache groups and recurrent external
profiles. SGLang's complete released-engine hybrid lifecycle remains open.
Follow [hybrid recovery contracts](hybrid-recovery.md) and the
[S5 implementation sequence](completion-plan.md#s53--public-lifecycle-and-hybrid-state-contracts)
before enabling these models. Historical fork controls are evidence for their
original revisions only.

## MLA and auxiliary state

DeepSeek-V4.1-Flash needs native kernel/topology readiness plus complete compressed,
indexer and auxiliary-state recovery. An earlier MLA byte control does not qualify
this model. Establish native output first, complete external recovery next, then
performance with the actual topology and precision. Unknown draft/auxiliary state
remains unsupported.

## Static LoRA

The accepted profile uses immutable local PEFT artifacts, startup content
fingerprints and effective LoRA settings in the cache identity. Stop the engine
before changing artifacts. Dynamic add/remove/reload, live weight replacement,
resumable sessions and caller aliases that bypass matching are not supported.
See [adapter identity](adapters.md) for effective SGLang LoRA configuration and
[the installed test](../python/tests/README.md#model-and-feature-gates).

## Adding a model

Declare every required state family and byte layout in the existing adapter and
Rust state contracts. Reject unknown state before connecting to the cache. Use
existing correctness gates for cold/partial/full hits, native HBM, restart and
output; add cancellation/page-reuse coverage when changing lifecycle behavior.
Record model revision, engine, precision, representation, topology and execution
mode in the support table only after the matching profile passes.

Historical native controls, Qwen3.5/Qwen3.8 and MLA experiments, their failures
and external artifact locations remain in the
[immutable model report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/models.md).
