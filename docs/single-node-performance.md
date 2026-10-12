# Single-node cache measurements

The historical matched serving cohort compares native HBM, native CPU offload, OrbitKV and LMCache MP with its recorded engine/runtime and budgets. OrbitKV improves recovery over HBM-eviction recomputation; native CPU offload remains faster in that workload. Resident HBM is a separate case. These results do not establish current-release or larger-model superiority.

The [immutable full report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md) retains tables, exact commands,
artifact locations, failed controls and uncertainty. Use [performance methodology](performance.md)
for new comparisons and [compatibility](compatibility.md) for current support.

## Deterministic comparison controls

See the corresponding section of the [frozen report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md).
Its original workload, revision and qualification limits remain unchanged.

## Engine-local Restore serving qualification

See the corresponding section of the [frozen report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md).
Its original workload, revision and qualification limits remain unchanged.

### Matched vLLM end-to-end comparison

See the corresponding section of the [frozen report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md).
Its original workload, revision and qualification limits remain unchanged.

### Measurement evidence and reproduction

See the corresponding section of the [frozen report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md).
Its original workload, revision and qualification limits remain unchanged.

## Historical compatibility and latency experiments

See the corresponding section of the [frozen report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md).
Its original workload, revision and qualification limits remain unchanged.

## Reproduce the latency experiment

See the corresponding section of the [frozen report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md).
Its original workload, revision and qualification limits remain unchanged.

## Single-node optimization order

See the corresponding section of the [frozen report](https://github.com/feichai0017/orbitkv/blob/251f463d8997eff783871ab2820da83a13a97196/docs/single-node-performance.md).
Its original workload, revision and qualification limits remain unchanged.
