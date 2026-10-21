# OrbitKV benchmarks

Run commands from the repository root. Keep raw measurements, logs, manifests,
plots and failed controls in an external artifact directory. Application
correctness gates live in `python/tests/`; tests here protect benchmark validity.

## Choose a benchmark

| Directory | Purpose | Main entry points |
| --- | --- | --- |
| [serving/](serving/README.md) | Official engine/backend comparisons, pressure and concurrency | `benches.serving.single_node`, `shared_manager_pressure`, `native_model_readiness` |
| [transfer/](transfer/README.md) | Local DMA/kernel copies, shapes and observation cost | `benches.transfer.communication`, `kernel_pairs`, `copy_observation` |
| [distributed/](distributed/README.md) | Remote cache recovery, TENT and RDMA | `benches.distributed.shared_cache_serving`, `rdma_profile`, `tent_trace` |
| [cluster/](cluster/README.md) | Reusable metadata churn and visibility | `benches.cluster.metadata` |
| [storage/](storage/README.md) | SSD paths, GDS prerequisites and storage codecs | `benches.storage.gds`, `storage/native/cpu_codec.rs` |
| [common/](common/README.md) | Process ownership, launch, workloads, metrics and reports | Imported by the benchmark families |

CUDA/C++ and Rust fixtures live in their family's `native/` directory. CPU harness
tests mirror the families under `tests/`. Shared test inputs belong in fixture
modules, separate from executable test cases. New files should extend an existing
owner and family, rather than add another root-level program or test suite.

## Run

```bash
.venv/vllm-release/bin/python -m benches.serving.single_node \
  --engine vllm --backend orbitkv --model /path/to/model \
  --output /var/tmp/orbitkv-bench/run-001

python -m pytest -q benches/tests
python -m pytest -q benches/tests/distributed
```

The CPU suite runs in CI without Torch, engines, CUDA execution or a Manager.
Selected C++ loader controls require a compiler and CUPTI headers. Installed
Uvicorn process controls are documented in [shared support](common/README.md).
Hardware benchmarks are explicit entry points, not part of default test discovery.
Run only the family tests affected by a local edit; run the complete CPU suite for
shared harness changes and before merging this layout migration.

Frozen external campaigns retain their original source revision and commands.
Use that frozen source when replaying old evidence; current commands use the
module paths above. Module moves do not qualify a new artifact or change a result.
Use [the completion plan](../docs/completion-plan.md) as the only delivery queue
and [performance](../docs/performance.md) for reviewed results and limitations.

Completed S2-only capacity/isolation/observer and three-host campaign drivers,
their dedicated tests and duplicate report pages are retired. Their
[immutable source](https://github.com/OrbitInfra/orbitkv/tree/dc52461078bb22910ba6c783ecf0d9213599b10b/benches/cluster)
and original freezes preserve reproduction. Current correctness tests remain
with the Rust owners and under `python/tests/`; failed qualifications stay failed.
