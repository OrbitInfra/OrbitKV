# Model recipes

A model loading in an inference engine does not establish external-cache
recovery. OrbitKV must restore every state component the engine reads, at a
legal boundary and with matching artifact, computation and layout identity.
The adapter pins are vLLM 0.33.0 and SGLang 0.5.23. All current OrbiTE GPU
and installed profiles remain unqualified.

## Choose a model

Start with dense Qwen3-8B or Qwen3-32B, TP=1/PP=1, eager execution and raw KV.
The existing recipes use these immutable revisions:

| Checkpoint | Revision |
| --- | --- |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen3-32B | `9216db5781bf21249d130ec9da846c4624c16137` |

These are reproducible inputs, not new qualification. [Historical model evidence](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/models.md)
retains native-only controls, original releases, failures and limits. Current
[compatibility](compatibility.md) takes precedence over a family name.

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

## Hybrid attention and recurrent state

Full, window and checkpoint recovery requirements are implemented in the shared
state contract. Each adapter must supply all required groups and a legal boundary;
a prefix hit count cannot replace missing state. vLLM V1 admits exact Full + SWA
groups with equal block sizes, atomic recovery and pinned Save jobs. Recurrent,
zeroing, unknown specs and MLA multi-group remain rejected. Its
[V2 opt-in](adapters.md#released-v2-lifetime-contract) is
limited to uniform Full attention, not hybrid recovery. SGLang also admits static
Mamba conv/SSM checkpoints through its released component, without ReplaySSM,
speculation, int8 checkpoints or side-state buffers. The next hardware input is an immutable Gemma 3 Full/SWA checkpoint, eager
TP=1/PP=1, raw KV, without speculation. Record its exact revision; require both
groups in engine logs, native-output parity, restart hits, per-group bytes and
[HMA faults](fault-qualification.md#fullswa-gates). This gate is written only,
unrun and unqualified. Follow [recovery contracts](hybrid-recovery.md) before
broadening those guards.

For SGLang's checkpoint path, the queued model gate uses an immutable
Qwen3-Next-80B-A3B checkpoint (Full + Gated DeltaNet state), eager TP=1/PP=1,
with its revision recorded. The [component gates](fault-qualification.md#sglang-component-gates)
require actual recurrent registration, output parity, restart hits, exact
conv/SSM fixture bytes and ownership faults. No model-family name is a qualification.

## MLA and auxiliary state

Ordinary SGLang `MLATokenToKVPool` latent/RoPE rows are admitted per rank.
The queued model gate uses an immutable DeepSeek-V2-Lite checkpoint, eager
TP=1/PP=1, with its revision recorded. DSA indexers, draft/auxiliary buffers and
separate scales remain rejected. Deduplicating replicated MLA state across
ranks remains M3 work despite the released writer-selection rule. A byte-layout
fixture is not model recovery. Validate native output, then complete external
recovery, before measuring performance.

## Static LoRA

Immutable local PEFT artifacts, startup fingerprints and effective LoRA settings
participate in cache identity. Stop the engine before changing artifacts. Dynamic
add/remove/reload and live weight replacement are unsupported. See
[adapter identity](adapters.md#immutable-static-lora) and
[installed tests](https://github.com/OrbitInfra/OrbitKV/blob/main/python/tests/README.md#model-and-feature-gates).

## Adding a model

Describe every required state family and layout in the existing adapter and Rust
contracts. Reject unknown state before connecting. Check cold/partial/full hits,
native HBM, restart, output, cancellation and page reuse. Record exact model and
engine revisions, precision, representation, topology and execution mode only
after the matching installed gate passes.
