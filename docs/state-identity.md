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
DRAM, SSD and the remote directory. The namespace is the complete storage
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

## Target contract

1. **Name the computation.** Each adapter supplies a versioned model identity
   from immutable weight revision/content, tokenizer and multimodal processor,
   active adapters/LoRA or other request-specific parameters, and the cache
   implementation variant. A deployment operator may supply a stable digest
   when hashing large weight files at startup is impractical. The adapter must
   fail closed if a required identity cannot be established.
2. **Name the bytes.** Serialize a versioned, domain-separated `StateKey`
   containing the computation identity, framework prefix/content digest,
   `[start, end)` token range, component and cache group, and the full
   `StateFormat`: dtype/quantization, block size, layout and strides, head
   shape, parallel rank/shape, and implementation compatibility. Use the same
   key in Query, Publish, DRAM/SSD indexes, and remote directory records.
   Re-keying the cache before 1.0 can invalidate old entries rather than
   carrying a compatibility path. Equal model names alone never authorize
   cross-framework byte reuse.
3. **Prove recovery.** A validator checks that a `StateBundle` covers every
   requested token without gaps or overlap, all components agree on model and
   format, and the model's recovery rule requires exactly those components at
   that boundary. Dense attention KV, MLA, sliding-window attention, recurrent
   checkpoints, DSA, draft, and auxiliary state need explicit capability
   definitions. A partial hit can only shorten prefill to a boundary the
   validator proves restorable; otherwise the engine recomputes.
4. **Fence page reuse.** The adapter sends a registration/session epoch and a
   generation-qualified page reference for every GPU source and destination.
   The manager validates these before copy and holds the relevant leases until
   CUDA completion. A stale page ID must never address a newly reused GPU slot.
5. **Expose capability, not guesses.** vLLM maps cache groups and HMA boundary
   state to the shared bundle contract. SGLang maps RadixAttention page and
   lifecycle events to that contract without maintaining a second radix tree.
   Unsupported components remain rejected rather than treated as attention KV.

The first integration should use the existing vLLM and SGLang native hash
algorithms within a versioned domain, then test whether a common token/content
digest can safely support cross-framework sharing. A matching text prompt is
insufficient evidence: byte-level compatibility needs an explicit format and
implementation check or a conversion path.

## Order and acceptance gates

The [completion plan](completion-plan.md) owns delivery order and qualification.
Identity and declared recovery validation are implemented; allocator-issued page
generations, complete model capabilities and physical fault reclamation retain
separate gates. The [original staged design](https://github.com/OrbitInfra/OrbitKV/blob/1f30895a0501670cabe16e64eeb5688e05a2aa54/docs/state-identity.md#order-and-acceptance-gates)
is historical, not another implementation queue.
