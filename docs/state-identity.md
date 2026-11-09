# Model-aware state identity and recovery

Correct KV reuse requires more than a matching prefix hash. A hit must name
the same model computation, identify compatible bytes, cover the requested
token range, and contain every component required to resume at that boundary.
OrbitKV must reject an ambiguous candidate and let the engine recompute it.

## Current implementation

Both adapters resolve a versioned SHA-256 computation identity at startup.
Local model, tokenizer, processor and model-code artifacts are fingerprinted by
content; identical artifact copies can keep the same identity after relocation.
Each initialization reads the content again. File timestamps are not used to
reuse an earlier digest: an equal-sized replacement can have the same timestamp
on a filesystem with coarse clocks. This work stays outside the request path.
Hub deployments require a full immutable commit revision. For large models,
`ORBITKV_MODEL_FINGERPRINT` accepts a lowercase 64-digit deployment digest and
skips artifact reads: the operator owns the accuracy of that assertion. It
must cover weights, tokenizer, processor and model code. Configuration still
participates in the identity even with an explicit artifact digest.

The identity also includes the engine release, computation settings and cache
representation. vLLM includes its HF configuration, quantization, attention and
kernel configuration, resolved KV dimension order, cache configuration hash,
hash algorithm/seed, dtype, block sizes and parallelism.
SGLang includes weight version, model overrides, quantization, attention
backend, rank, dtype, page size and buffer shape/strides. The Cache Manager
binds this identity to the actual registered storage slots, groups, segment
geometry and page layout when registration seals. GPU addresses and pool
capacity do not affect the stored representation. Workers and the scheduler
session must agree on identity and topology for a given instance.

`orbitkv-state::StateKey { namespace, hash }` is now the index type used by
DRAM, SSD and the peer inventory index. The namespace is the complete storage
identity; the hash is a versioned, length-framed native chained prefix hash and
cache group, including group zero. Query and Publish derive the same key from
the registered instance. Hashing model files and storage layouts stays out of
the per-block hot path. Old raw-hash keys and process protocol versions are
invalidated; there is no compatibility lookup or old client alias.

SGLang and vLLM hybrid adapters compile groups into `RecoveryContract` rules and validate
`StateBundle` evidence on live queries. Prefixes require contiguous pages; windows
require the complete rounded trailing range; recurrent/conv state requires an
exact checkpoint at the selected end. The bridge obtains the absolute origin
from the engine's valid HBM prefix and matches each group against leased page positions.
It never numbers a partial transfer from zero. Model/layout isolation comes from
the registered namespace; the engine remains responsible for declaring all required
state and holding the base prefix valid. This is a recovery-contract check, not
a formal proof of the model's mathematics. See [hybrid recovery](hybrid-recovery.md).

vLLM compiles full-attention, sliding-window and aligned recurrent groups;
hybrid P/D-tail restrictions remain. `StateDescriptor` and `StateFormat`
remain planning types; their fields are not a new per-page wire
protocol. Publish still uses raw engine block IDs without generation checks.
Cross-engine reuse, dynamic LoRA and live weight updates are unsupported.
Immutable local static LoRA follows the [declared adapter-set contract](adapters.md#immutable-static-lora);
weight changes require restart and a new artifact identity. Engine-specific hashes remain in separate identity domains.

`ORBITKV_CACHE_SCOPE` optionally separates tenants or experiments within the
same deployment. It never replaces the model fingerprint. Both engines use
these same environment variables; inference processes still connect to their
Cache Manager over UDS/iceoryx2 regardless of the tier holding the bytes.

## Page reuse and unsupported state

Registration/session epochs do not identify every reuse of a GPU slot. The
selected engine handoffs lack allocator-issued page generations, and Publish
still uses raw engine block IDs. Closing this gap requires consumed engine
contracts and actual reuse/fault tests; a local counter cannot prove safety.
Unknown auxiliary state and cross-engine byte conversion remain unsupported.

## Order and acceptance gates

The [completion plan](completion-plan.md) owns delivery order and qualification.
Identity and declared recovery validation are implemented; allocator-issued page
generations, complete model capabilities and physical fault reclamation retain
separate gates. The [original staged design](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/state-identity.md#order-and-acceptance-gates)
is historical, not another implementation queue.
