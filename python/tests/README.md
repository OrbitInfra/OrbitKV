# Python tests

Run the gate for the boundary you changed. Default tests require no GPU,
Torch, inference engine, native build or running Manager. Hardware checks are
explicit integration/E2E/release gates; a missing runtime leaves them unqualified.

## Layout

| Directory | Responsibility |
| --- | --- |
| `unit/` | Python adapter/configuration and test-support contracts |
| `integration/` | Native client, Manager, CUDA bytes and lifecycle boundaries |
| `e2e/` | Official vLLM/SGLang output, external hits, restart and P/D composition |
| `release/` | Installed wheel, shared Manager and opt-in feature profiles |
| `stress/` | Pressure, concurrency and fault workloads |
| `support/` | Shared process, runtime, GPU identity and installed-artifact helpers |
| `../../benches/tests/` | Reusable benchmark validity; no inference/GPU execution |

Keep a test only when it protects a current developer, CI, release or operational
consumer. Merge cases that protect the same contract; keep assertions about real
behavior rather than source spelling or obsolete campaign bookkeeping. Keep
heavy imports inside fixtures/functions so default collection stays source-only.

## Default unit gate

From `python/`:

```bash
uv run --isolated --no-project --with pytest --with numpy \
  --with 'requests>=2.26.0' --with 'tomli>=2.0; python_version < "3.11"' pytest
```

Markers in `pyproject.toml` exclude integration, E2E, stress and release by
default (`release_smoke` selects installed-artifact checks). Stub modules must not shadow
real runtime modules during heavy-gate collection.

`--no-project` keeps this command independent of Maturin and optional engine
dependency resolution. CI uses the same source-only dependency boundary.

From the repository root, benchmark-only changes use the affected family:

```bash
python -m pytest benches/tests/transfer
```

Use the complete `benches/tests` suite for shared harness changes or before
merge. Retired S2 campaign controllers/tests are reproduced from their
[immutable source](https://github.com/feichai0017/orbitkv/tree/dc52461078bb22910ba6c783ecf0d9213599b10b/benches/cluster).

## Select a hardware gate

Run from `python/` in the selected installed engine environment. Set
`ORBITKV_CACHE_MANAGER_BINARY` to the matching frozen Manager and
`ORBITKV_MOONCAKE_LIB_DIR` where required. Actual CUDA/IPC and model prerequisites
are part of each gate, not provided by pytest collection.

| Change | Gate |
| --- | --- |
| Server/native/session or GPU registration | `pytest -m integration` with a matching built/installed runtime |
| Distributed membership, inventory or peer recovery | `ETCD_BIN=/path/to/etcd pytest -m integration tests/integration/test_distributed_cache.py` |
| vLLM adapter/save/load/recovery | Official vLLM 0.31.0: `python -m pytest -m e2e tests/e2e/test_vllm_e2e_correctness.py --model /path/to/model --max-model-len 4096` |
| SGLang linker/layout/plugin | Official SGLang 0.5.22: `python -m pytest -m e2e tests/e2e/test_sglang_direct_e2e.py --model /path/to/model` |
| Shared-cache adapter or peer ownership | `pytest -m e2e tests/e2e/test_shared_cache.py -k vllm --model /path/to/model`; repeat with `-k sglang` in its environment |
| Warm-hit concurrency | `pytest -m stress tests/stress/test_vllm_warm_hit_stress.py --model /path/to/model` |
| Native P/D composition | `tests/e2e/test_vllm_native_pd_e2e.py` or `test_sglang_pd_e2e.py`, with `-m e2e` and the [native P/D runtime](../../docs/pd.md) |

The shared-cache serving gate uses two replicas/two Managers on one host over
TCP; it does not qualify cross-host HA, RDMA or physical ranks. Installed CUDA
identity checks require the declared physical devices. Run `pytest --help` for
layout, SSD, copy backend, process-trace and deployment options rather than
adding another launcher.

## Installed runtime and lifecycle

Use a tests-only projection or installed-test environment when validating a
wheel. Do not put source `python/orbitkv` ahead of installed site-packages.
`support/installed_pytest.py` and shared runtime helpers record the actual
package/native origins, Manager and engine process identities.

Keep cleanup ownership before startup. Normal SIGTERM, actual parent waits,
remaining owned workers and forced cleanup are separate observations. Preserve
original errors and complete logs. A service leader exiting zero is not proof
that every native worker drained or physical DMA ended. Unknown ownership must
not be released by a test timeout.

The vLLM helper uses the official shutdown grace period; SGLang cleanup checks
its process group. Current scoped healthy results do not erase previous native
SIGKILL/IPC warnings or establish fault reclamation. [S3](../../docs/completion-plan.md#s3--transfer-lifetime-and-generation-safe-ownership)
and [compatibility](../../docs/compatibility.md) retain the actual limits.

`support/registration_campaign.py` remains the installed registration controller
and shared runtime-evidence owner used by current gates. It is not production
Python. Run its CPU units when changing admission/runtime checks; do not build a
second controller or manually weaken frozen cell validators.

## Release smoke

Install the complete wheel outside the checkout, with the matching Python/CUDA
runtime. Check native import, packaged Manager/TENT libraries and actual CUDA
before claiming an installed engine result:

```bash
python -m pytest -m release_smoke tests/release/test_installed_wheel.py \
  -k vllm --model /path/to/immutable-model
```

Follow [package releases](../../docs/releases.md) for the full artifact matrix,
licenses, loader requirements and publication. A source test or CPU-native
fixture does not qualify the final CUDA wheel.

### Two engines sharing one Manager

Use `tests/release/test_shared_manager.py` with `-m release_smoke`,
`--vllm-release-python /path/to/vllm/bin/python`,
`--sglang-release-python /path/to/sglang/bin/python`, one immutable `--model`
and the same installed Manager. Select explicit direct/kernel and DRAM/io_uring
profiles. The current 900-second H20
dense eager acceptance establishes progress, bytes and bounded resources for
its frozen configuration; fairness, tail isolation, Manager restart and separate
containers remain open. [Deployment](../../docs/deployment.md) owns the recipe.

## Model and feature gates

- Static immutable LoRA: `tests/release/test_static_lora.py` in each engine
  environment; dynamic mutation/live weights are unsupported.
- Explicit FULL decode Graph: run `python -m pytest -m release_smoke
  tests/release/test_installed_wheel.py -k vllm --model /path/to/model
  --release-cuda-graph`; repeat with `-k sglang` in its environment. Ordinary
  E2E modules do not consume this flag. Current acceptance is batch-1/eight-token
  dense TP1/PP1, not arbitrary graph shapes or hybrid replay.
- Storage encoding: use the relevant integration/release codec gate and exact
  representation/quality controls. FP8 weights and FP8 KV are different inputs.
- TP/shards/P/D: run their explicitly selected gate; one-node CPU controls do
  not qualify physical multi-rank serving or partial-write/cancellation faults.

Choose models from [model recipes](../../docs/models.md). Preserve native HBM,
cold miss, partial/full external hits, bytes and output comparisons. Do not treat
native output-only controls or historical fork experiments as current recovery.

## Evidence and previous recipes

Raw results, logs, traces, manifests and per-run reports belong outside the
checkout. Freeze consumed artifacts, retain failed runs and report uncertainty
before claiming performance. Use [benchmark methodology](../../docs/benchmark-evidence.md)
and the existing gate's workload instead of adding another verification layer.

The [pre-cleanup gate reference](https://github.com/feichai0017/orbitkv/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/python/tests/README.md)
retains exact earlier installed campaign commands and evidence locations. Its
recorded versions and failures remain unchanged; the
[completion plan](../../docs/completion-plan.md) is the only execution queue.
