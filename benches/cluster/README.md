# Metadata benchmark

The maintained `benches.cluster.metadata` benchmark measures full-Manager
inventory churn and visibility. It uses real etcd and the existing integration
helpers; it does not build artifacts or run inference.

Run from a tests-only projection root containing `benches/` and `tests/`, with
the matching wheel installed and without a source `orbitkv/` package:

```bash
/path/to/engine/bin/python -m benches.cluster.metadata \
  --profile 2 --output /var/tmp/orbitkv-bench/metadata-001
```

Use the matching installed runtime/test projection, frozen Manager/etcd/native
libraries and actual GPU prerequisites. Run `--help` for the selected workload.
See [distributed cache](../../docs/distributed-cache.md) for current architecture
and [compatibility](../../docs/compatibility.md) for support limits.

Completed S2 capacity, isolation/observer and three-host campaign controllers,
analysers and their dedicated tests are retired. Reproduce an old campaign from
[its immutable source](https://github.com/feichai0017/orbitkv/tree/dc52461078bb22910ba6c783ecf0d9213599b10b/benches/cluster)
and original artifact/freeze, not a current checkout. The
[frozen plan](https://github.com/feichai0017/orbitkv/blob/dc52461078bb22910ba6c783ecf0d9213599b10b/docs/completion-plan.md)
retains results and evidence locations, including failed qualifications.

Current distributed correctness gates remain under `python/tests/integration/`
and the relevant Rust crates. Restore a historical controller only when a new
operational consumer actually needs it; do not recreate the old campaign framework
for routine S4/S5 work. Retiring a controller does not pass its failed gate.
