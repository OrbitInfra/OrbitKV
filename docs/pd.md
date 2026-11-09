# Prefill/decode transfer

> OrbiTE is now the only transfer engine. GPU/RDMA/GDS and installed profiles
> require fresh hardware qualification. Earlier measurements retain their original
> revisions and backend (TENT where recorded); CPU/TCP and io_uring checks are CI
> evidence, not installed GPU qualification.

OrbitKV uses **official vLLM 0.33.0 and SGLang 0.5.23**. Engines own live
P/D routing, allocation, transfer and release. OrbitKV independently caches
completed state. Manager-to-Manager sharing continues to use Rust/OrbiTE; live
P/D does not require the same transport.

Native P/D plus OrbitKV is a **qualification candidate**, not a generally
supported deployment. The current status and remaining fault gates are in
[the completion plan](completion-plan.md#milestones).
Ordinary cache serving does not require NIXL, a P/D router, or an engine patch.

| Responsibility | vLLM | SGLang |
| --- | --- | --- |
| P/D lifecycle and transfer | Released `NixlConnector` | Released disaggregation queues and selected native backend |
| Router | `vllm-router==0.1.15` | `sglang-router==0.3.2` |
| Historical cache | `OrbitKVConnector` via `MultiConnector` | `UnifiedRadixCache` + external `OrbitKVLinker` |
| Shared cache between Managers | Rust/OrbiTE | Rust/OrbiTE |

The custom P/D connectors, handshake/proxy, partial-tail extension and fork-only
TENT payload adapters are removed. Native P/D carries live prompt tails;
the independent cache stores complete engine blocks. There is no alternate
OrbitKV P/D request owner or maintained engine fork.

## vLLM NIXL and MultiConnector

This follows the separation used by the
[LMCache 0.5.5 MP recipe](https://github.com/LMCache/LMCache/blob/05a013b29da78cf2321b9b46ec5039dde2fb0bb0/docs/source/mp/disaggregated_prefill.rst).
The required real-block allocation fix, vLLM #46865, is in 0.30.0. That does not
transfer LMCache's deployment qualification to OrbitKV.

Install the official vLLM release, its `nixl==1.4.1` dependency (including the
matching CUDA backend), the router, and the OrbitKV wheel. Use
`VLLM_USE_V2_MODEL_RUNNER=0`: the released V1 runner drains preempted saves before
page updates. OrbitKV rejects V2 and multi-group/recurrent models until their
required lifecycle boundaries have a safe released implementation.

The candidate profile uses a read/write cache on P and a save-only cache on D:

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {"backends": ["UCX"]}
      },
      {
        "kv_connector": "OrbitKVConnector",
        "kv_connector_module_path": "orbitkv.vllm",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
          "orbitkv.bootstrap_socket": "/tmp/orbitkv-decode.sock",
          "orbitkv.mode": "save_only"
        }
      }
    ]
  }
}
```

For P, set the NIXL role to `kv_producer`, put the cache child first, select
`read_write`, and provide its local Manager socket. Let vLLM generate fresh
engine IDs. Give each process a distinct `VLLM_NIXL_SIDE_CHANNEL_PORT` and an
appropriate `VLLM_NIXL_SIDE_CHANNEL_HOST`.

The [local launcher](https://github.com/OrbitInfra/OrbitKV/blob/main/scripts/run_pd_local.sh) takes `PREFILL_CACHE_SOCKET`,
`DECODE_CACHE_SOCKET`, `VLLM_PYTHON` (default `.venv/vllm-release/bin/python`),
`PREFILL_GPU` and `DECODE_GPU`. It starts official vLLM and `vllm-router` on two
local GPUs. It does not start Managers or install engines. Separate Managers
are recommended for capacity isolation; sharing one needs a contention gate.
NIXL/UCX chooses the data transport. Do not label a run TCP or GPUDirect RDMA
without recording its actual backend and devices.

D's cache never queries or restores in this profile. Native NIXL owns incoming
pages while both children may retain completed pages for sending/saving.
`MultiConnector` aggregates their completion before block release. P cache
publication remains best effort and cannot weaken reliable P/D delivery.
Read/write cache selection on D, both connector orders, concurrent requests,
restart and pressure are separate test candidates, not default guarantees.

## SGLang native P/D

Use the official engine and router with its native transfer dependency. The
current candidate exercises `--disaggregation-transfer-backend mooncake` with
the released `mooncake-transfer-engine` package, independently of OrbitKV's OrbiTE sharing.
No transfer-engine class substitution, factory patch or fork callback is used.

Add `--radix-cache-backend orbitkv`, `--enable-unified-cache-external-linker`
and `ORBITKV_SGLANG_ENDPOINT=unix:///path/to/local-manager.sock` to each worker.
Select `--disaggregation-mode prefill` or `decode`; D also needs
`--disaggregation-decode-enable-radix-cache`. The native router owns bootstrap
and routing. P restores historical state, D only publishes completed state;
`RecoveryLinkerWrapper` disables external loads on D. Host-pool retraction,
DSA/draft/unknown auxiliary layouts remain rejected.

Enable `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE=1` on both P and D
for the candidate; leave its timeout at 30 seconds. In official 0.5.23 this
protection defaults off. Even when enabled, the native queue can release device pages
on timeout without a full drain acknowledgement. Therefore in-flight transfer
cancellation, peer loss and delayed-ACK page reuse remain unsupported by this
candidate; enabling the flag does not qualify them. The model test cancels an
already-running decode request and does not claim to exercise this DMA boundary.
See the [release audit](adapters.md#remaining-internal-hooks).

The ordinary cache has two version-coupled internal Hooks, plus optional queue
preparation. Their purpose and removal conditions are documented in
[the adapter audit](adapters.md#remaining-internal-hooks).
They are not P/D lifecycle Hooks. Native P/D cancellation/retraction correctness
must be established against the release itself; the earlier fork's drain/ACK
fixes are not assumed present.

## Qualification and reproduction

Use a new external runner containing only `python/tests/` and
`python/pyproject.toml`, with a frozen installed wheel and clean official engines.
Run from that runner's `python/` directory. Do not add the source `python/orbitkv`
package to `PYTHONPATH`: it can shadow the verified native wheel. Set
`ORBITKV_CACHE_MANAGER_BINARY` to the Manager executable in the selected installed
wheel and create the parent of each `--basetemp` before running:

```bash
python -m pytest -m integration tests/integration/test_vllm_connector_contract.py
python -m pytest -m e2e tests/e2e/test_vllm_native_pd_e2e.py -k save-only \
  --model /path/to/dense-model --basetemp /var/tmp/orbitkv-native-pd/vllm-001
python -m pytest -m e2e tests/e2e/test_sglang_pd_e2e.py \
  --model /path/to/dense-model --basetemp /var/tmp/orbitkv-native-pd/sglang-001
```

Model gates compare exact output with monolithic controls, require cache bytes
and restart reuse, and exercise observed preemption or retraction. These are
separate from delayed-ACK, partial-submit, cancellation during DMA and page-reuse
fault gates. A successful output test alone does not qualify those failures.
No cross-host HA, GPUDirect RDMA, heterogeneous rank/GPU, hybrid P/D or native
GDS claim follows from same-host dense-model evidence.

Historical model/fault attempts, original revisions and unaccepted scopes are
preserved in the [frozen P/D record](https://github.com/OrbitInfra/OrbitKV/blob/4a36b11c680b9e82f05a663da5cb38c0dc694a9b/docs/pd.md#qualification-and-reproduction).
They do not qualify current OrbiTE cache composition.
