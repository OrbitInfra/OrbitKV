# Released engine interface audit

The selected source/package pins are official **vLLM 0.31.0**
(`db9527a46873454610df6dbedf79a36d6bf1a7f6`) and **SGLang 0.5.22**
(`baa9befa381511cd4d56f83980951e5a45fffc6d`). The SGLang migration passes
official installed CPU and Qwen3-8B dense DRAM/io_uring byte/model/restart controls.
Earlier accepted 0.5.21 workloads retain their original scope. The
[completion plan](completion-plan.md#s5--released-engine-integration-and-upstream-contributions)
is the execution queue; this page describes consumed contracts.

## Official runtime boundaries

| Engine | Consumed interface | Ownership and current limit |
| --- | --- | --- |
| vLLM | KV Connector factory, scheduler allocation/lookup/save metadata, worker registration/layer waits/preemption/results | Engine owns GPU pages; V1 and one attention cache group required |
| SGLang | Package plugin, public radix backend registration, UnifiedRadixCache and external linker | Native tree owns private load slots, locks, insertion and abort cleanup; Full/SWA only |
| vLLM P/D | Official NixlConnector + MultiConnector and upstream router | NIXL owns live handoff; OrbitKV independently stores completed state |
| SGLang P/D | Official disaggregation queues/backend and router | Native transport owns handoff; cache restores on P and saves on D |

No engine fork, installed-file patch, custom P/D connector, handshake or proxy is
required. Different engines do not automatically share compatible KV bytes.

## SGLang 0.5.22 load completion

The released
[`finish_layer_wise_loading(index) -> bool`](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py)
callback runs after forward, before loaded destinations can be inserted or
released. OrbitKV waits for the whole native Restore and returns success only
when every group has landed. A never-admitted cancelled batch may return false;
unknown submission, timeout or lost completion raises while owners remain held.
Logical failure and process disappearance do not prove physical completion.

The adapter delegates private destination allocation, tree insertion and native
abort cleanup to the released wrapper. The old completed-load polling, custom
`_update_load` implementation and recurrent component override are removed.
Boundary/rank translation remains in `RecoveryLinkerWrapper`. Requests and match
results use released `prefix_len` and `device_prefix_len`; request refresh uses
`refresh_fill_ids()`. Source CPU controls are not installed model/GPU evidence.

Full and SWA are the released external-linker components. Recurrent/SSM,
DSA/draft/auxiliary state and unsupported request rings are explicitly rejected
until their complete recovery contract is consumed. Generic layout bytes do not
qualify model output or page reuse.

## Remaining internal Hooks

The public Hook registry does not make string targets stable APIs. Keep each
correctness protection until a released replacement is actually consumed.

| Target | Purpose | Removal gate |
| --- | --- | --- |
| `TpModelWorker.init_cuda_graphs` BEFORE | Install stable restore events before Graph capture; cache factory runs later | Released worker preparation callback plus actual eager/Graph restore and page-reuse gates |
| `PrefillAdder.add_one_req` AROUND | Defer pending SSD/peer lookup and preserve rank agreement | Released asynchronous lookup/admission contract plus miss/hit/cancel/rank gates |
| `Scheduler._add_request_to_queue` AROUND, opt-in | Optional queued preparation/warming | Released enqueue callback or remove optimization with matched workloads |

The former Scheduler abort Hook, runner monkey patch and private P/D observers
are removed. Plugin discovery stays lazy and opens no native/GPU resources for
an unselected backend.

## Upstream work and independent limits

- [vLLM #59410](https://github.com/vllm-project/vllm/pull/59410) tracks V2 recurrent
  preparation ordering. Official 0.32.0 still needs this contract; the selected
  runtime remains V1/one group. A contribution does not extend support.
- SGLang worker preparation [#44856](https://github.com/sgl-project/sglang/pull/44856)
  and pending admission [#44873](https://github.com/sgl-project/sglang/pull/44873)
  are proposals, not consumed released APIs. Their contribution CPU evidence
  does not authorize removal of either safety Hook.
- SGLang's released device P/D timeout still lacks a full drain ACK. Host staging
  protection and ordinary cache completion do not qualify device P/D faults.
- vLLM teardown race [#48745](https://github.com/vllm-project/vllm/issues/48745)
  and [#49000](https://github.com/vllm-project/vllm/pull/49000) retain their original
  failure/quarantine. A normal API exit is not proof of physical drain.

Native P/D is a composition candidate. Qualify cancellation, peer loss, partial
submission, delayed ACK, restart and page reuse separately. Dynamic LoRA/live
weight mutation, hybrid recovery, additional ranks and Graph shapes also need
their own consumed lifecycle and installed model gates.

## Historical evidence

The [pre-migration audit](https://github.com/OrbitInfra/orbitkv/blob/d9d500b9f45230662121387a2d2d6d7ad9ac9eb2/docs/engine-release-audit.md)
retains prior release checks, contribution commits, accepted/failed controls and
external artifact locations. No historic failure or qualification is overwritten
by this interface migration. Current capability claims belong in
[compatibility](compatibility.md) and [model recipes](models.md).
