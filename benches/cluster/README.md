# Metadata and etcd benchmarks

Run commands from the repository root. [Benchmark navigation](../README.md).

### S2.10 metadata performance qualification

Build and freeze the Manager, wheel, server-test pressure source, extension and
TENT libraries before starting these gates. Build the metadata-only pressure
fixture with `cargo test --release --no-run -p orbitkv-server --no-default-features
--features cuda-13,mooncake,test-hooks` on the CUDA 13 profile. Freeze that test
executable separately; debug fixture CPU/RSS is not the formal production-profile
reference. The capacity harness selects one visibility endpoint per run:

```bash
python -m benches.cluster.live_store_capacity \
  --owners 16 --seed capacity-01 --samples 1000 --warmup-cycles 50 \
  --visibility-mode ordinary --enforce-thresholds \
  --output /external/s2-10/capacity-16-ordinary

python -m benches.cluster.live_store_capacity \
  --owners 16 --seed capacity-01 --duration-seconds 60 \
  --visibility-mode concurrent-barrier --barrier-concurrency 4 \
  --output /external/s2-10/capacity-16-concurrent
```

`ordinary` never calls the synchronization endpoints in its measured path.
Serial and concurrent barrier runs are diagnostics and cannot pass the ordinary
50 ms gate on its behalf.

Isolation uses one full foreground Manager and a test-owned Rust inventory source
from the frozen `orbitkv-server` test binary. Quiet and pressure runs have the
same topology, scope, foreground bytes, medium, cadence and GPU synchronization.
The pressure source changes metadata through the production inventory stream
without foreground GPU copies or SSD I/O:

```bash
python -m benches.cluster.live_store_isolation \
  --condition quiet --medium dram --seed pair-01 --warmup-rounds 50 \
  --samples 1000 --cadence-ms 1000 --order-index 0 \
  --output /external/s2-10/isolation/dram-pair-01-quiet

python -m benches.cluster.live_store_isolation \
  --condition pressure --medium dram --seed pair-01 --warmup-rounds 50 \
  --samples 1000 --cadence-ms 1000 --order-index 1 \
  --output /external/s2-10/isolation/dram-pair-01-pressure
```

Set `ETCD_BIN`, `ORBITKV_CACHE_MANAGER_BINARY`, `ORBITKV_MOONCAKE_LIB_DIR`
and `ORBITKV_SERVER_TEST_BINARY` to one frozen candidate. Reverse condition order
for alternating pairs. Use `benches.cluster.summarize_live_store_isolation` only after all
five pairs finish; request samples within a run are not independent repetitions.

S2.10a diagnostics reuse the same harness without changing the foreground path.
Pass a positive bounded event limit to use the explicit diagnostic client calls
and Manager/storage timeline. The pilot contract is non-qualifying:

```bash
python -m benches.cluster.live_store_isolation \
  --condition quiet --medium ssd --seed diagnostic-pilot-on \
  --warmup-rounds 10 --samples 120 --cadence-ms 250 --order-index 1 \
  --diagnostic-timeline-limit 16384 \
  --output /external/s2-10a/pilot/ssd-quiet-diagnostic-on

python -m benches.cluster.summarize_live_store_diagnostics \
  --run /external/s2-10a/pilot/ssd-quiet-diagnostic-on \
  --output /external/s2-10a/pilot/ssd-quiet-diagnostic-on/stage-summary.json
```

The limit requires transfer tracing and is set by the harness before importing
the native extension or starting the Manager. Zero leaves the new stages off.
`samples.jsonl` owns client submit/return observations; `diagnostic-events.jsonl`
owns bounded Manager/insert/SSD events keyed by session epoch/token and channel
request ID. Thread resource snapshots include schedstat when Linux exposes it.
Reaching `diagnostic_timeline_limit`, missing a required stage or comparing clocks
outside the recorded host domain invalidates the run. SSD queue/durability stages
remain after save return and are never folded into `manager_save_submit_ms`.

The S2.10b observer follow-up changes only where the same owner-status HTTP
poll, JSON parse and bounded JSONL record execute. `in-process` uses a Python
thread in the foreground harness; `helper-process` uses a child that imports no
CUDA, Torch or OrbitKV module. Both modes use one future monotonic epoch, fixed
observer/foreground phase offsets, every scheduled 25 ms slot without skipping,
and the same finite poll count. The ready/start/complete/stop/exit/error records
are mandatory: a helper error, digest mismatch or missing sample invalidates the
cell. A short tool smoke uses the formal cadence and pressure but is not a
performance result:

```bash
python -m benches.cluster.live_store_isolation \
  --condition pressure --medium ssd --seed observer-smoke-a \
  --warmup-rounds 10 --samples 20 --cadence-ms 1000 --order-index 0 \
  --observer-mode in-process --observer-sample-ms 25 \
  --observer-phase-offset-ms 0 --foreground-phase-offset-ms 12.5 \
  --diagnostic-timeline-limit 32768 \
  --output /external/s2-10b-observer/smoke-a

python -m benches.cluster.live_store_isolation \
  --condition pressure --medium ssd --seed observer-smoke-b \
  --warmup-rounds 10 --samples 20 --cadence-ms 1000 --order-index 1 \
  --observer-mode helper-process --observer-sample-ms 25 \
  --observer-phase-offset-ms 0 --foreground-phase-offset-ms 12.5 \
  --diagnostic-timeline-limit 32768 \
  --output /external/s2-10b-observer/smoke-b
```

The formal A/B run is a separately reviewed, pressure-only SSD cohort. Do not
pool request samples or use a smoke result for causality. Run
`benches.cluster.analyze_observer_isolation` only against the frozen eight-cell settings
and complete raw cells; its paired bootstrap unit is the independent A/B pair.
A positive result can justify a benchmark-harness change only and cannot
reclassify the failed isolation qualification.

Installed-artifact qualification runs the harness from a tests-only projection:
copy `benches/` to the projection root, copy `python/tests/` as `tests/`, and
retain the relevant pytest configuration without copying `python/orbitkv/`.
Put the installed wheel's site-packages before that projection in the driver
`PYTHONPATH`. `live_store_isolation` passes that installed site-packages path
explicitly to every Manager, including the namespace probe, so embedded Python
can import `orbitkv.client.gpu` from the wheel while the driver still imports
test support from the projection. A source checkout remains the default for
normal integration tests; installed qualification must record the client,
extension, Manager and helper paths and reject a source package that shadows the
wheel.
