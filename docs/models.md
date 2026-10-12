# Model qualification

Qualify the checkpoint revision, engine release, cache representation and execution
profile together. Loading weights or producing native output does not establish
complete external state recovery. Current pins are vLLM 0.31.0 and SGLang 0.5.21;
see [compatibility](compatibility.md) and [the interface audit](engine-release-audit.md).

## Current official-release scope

| Checkpoint | Immutable revision | Accepted scope | Remaining boundary |
| --- | --- | --- | --- |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | Scoped dense ordinary DRAM/io_uring recovery/restart on both official engines | Broader rank, fault, graph and performance profiles are separate |
| Qwen3-32B | `9216db5781bf21249d130ec9da846c4624c16137` | BF16 TP=1/PP=1 H20 ordinary DRAM/io_uring recovery; vLLM Graph NONE, SGLang's recorded default graph setting | Physical lifetime, broader topology and matched serving performance |
| Qwen3.8-27B BF16 | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Native output controls only on both official engines | Complete hybrid external recovery and performance |
| Qwen3.8-27B FP8 | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | Native output controls only, with BF16 KV, on both official engines | Complete hybrid external recovery, quality and performance |
| DeepSeek-V4.1-Flash | `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` | Official metadata/configuration checked | Native kernel/topology readiness and complete compressed/auxiliary-state recovery |

Model revisions are recorded as measured inputs, not recommendations to substitute
another checkpoint. Historical vLLM 0.29.0 / SGLang 0.5.20 recovery results below
retain those versions and do not reopen current guards. Engine-native FP8 KV and
FP8 weights are separate: see [storage formats](storage-formats.md).

## Native Qwen3.8 controls

Complete BF16 and FP8 input inventories have independent acceptance, including
weights, tokenizer/template, configuration and license files. Four frozen official
engine/model cells each complete six serial greedy requests at input lengths
128, 1,024 and 4,096, with 16 output tokens and two repeats: 24 requests total.
Repeated text/token IDs and finite selected-token log probabilities pass within
each cell. Both weight variants use BF16 KV in text-only eager TP=1/PP=1 inference.

This does not establish cross-engine or cross-precision equivalence, semantic
quality, HBM hits, external recovery or performance. Qwen3.8 has 16 attention and
48 GDN layers; the vLLM V2/multiple-group/recurrent external-cache guard stays on.
Eight available GPUs do not qualify physical TP=8 cache recovery.

All four API leaders exit zero and final owned resources are empty. Native vLLM
EngineCore force-kill/semaphore diagnostics and SGLang adopted-child -9 exits
remain lifetime limits; leader exit is not every-child drain proof. The original
missing-ninja startup failure and separately reviewed PATH correction remain
preserved. Use each engine's own frozen tool PATH rather than borrowing tools
from another environment or modifying official installations.

Exact inventories, inputs, output controls and provenance are in the
[immutable full model report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/models.md),
which retains external artifact/reviewer paths. Input acceptance, native output,
external recovery and matched performance remain distinct gates.

For an independently owned native service and frozen token-ID inputs:

```bash
uv run --isolated --no-project --with requests python -m benches.native_model_readiness \
  --engine vllm --url http://127.0.0.1:8000 --model /path/to/immutable-model \
  --prompts /path/to/frozen-token-ids.json --output-tokens 16 --repeats 2 \
  --output /var/tmp/orbitkv-native-readiness/vllm-001
```

Use `--engine sglang` with its native service. This program checks output; the
deployment owner must separately verify installed artifacts, leases and cleanup.

## Pretrained checkpoints

Dense external-cache recovery uses immutable artifacts, matching computation and
registered byte layout. Qwen3-32B's scoped ordinary gate passes fourteen checks
with two expected hybrid-only skips across both engines. Its full model/runtime
freeze and shutdown limits remain in the immutable report. Distinct engine cache
layouts are not automatically interchangeable.

Historical synthetic or fork hybrid controls validate declared ranges/checkpoints;
they do not qualify complete recovery on the current official releases.
See [hybrid recovery contracts](hybrid-recovery.md).

### Qwen3.5 after bounded Restore partitioning

Historical bounded-partition recovery and its model/precision limits remain in
[the frozen Qwen3.5 section](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/models.md#qwen35-after-bounded-restore-partitioning).
Do not transfer its hybrid support to official vLLM's currently rejected profile.

### Qwen3.8 on H20

Earlier hybrid/fork results remain in
[the frozen H20 section](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/models.md#qwen38-on-h20).
The official BF16/FP8 native controls above have separate inputs and acceptance;
neither qualifies external hybrid recovery.

### MLA controls on H20

Historical MLA bytes/output controls and their engine/layout constraints remain in
[the frozen MLA section](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/models.md#mla-controls-on-h20).
DeepSeek-V4.1-Flash's indexer, compressed and auxiliary state requires its own
complete released-engine contract; a preceding MLA control is not that gate.

## Larger hybrid targets

Qualify every required state family and legal boundary before external recovery.
Unknown auxiliary/draft/indexer state remains unsupported. Run native controls
first, complete recovery next, then matched native/OrbitKV/LMCache performance
with actual tokens, budgets, precision, execution modes and uncertainty.
The [completion plan](completion-plan.md#s53--public-lifecycle-and-hybrid-state-contracts)
owns implementation order. Historical larger-target metadata and open gates are
retained in the immutable full report; they are not deployment support claims.
