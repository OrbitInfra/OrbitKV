# Shared benchmark support

Run performance experiments from the repository root. Benchmark programs, reusable workloads and harness
tests live here. Generated measurements and reports belong outside the checkout. Runtime Python
code belongs in `python/orbitkv/`; correctness gates belong in `python/tests/`.

`single_node` accepts repeated `--resource-lock-fd` arguments from an external
controller. It forwards those already-open regular-file descriptors to both
the Manager and engine; the controller still owns admission, quarantine and
final physical cleanup. Compiler/JIT caches are private to each output cell.
After a completed workload or foreground failure, `single_node` stops the GPU
engine while the cache service remains available for context unregistration,
then stops that service. Failed startup follows service-local cleanup.
This order does not establish CUDA IPC retirement or native-child/crash drain.
Cleanup receipts retain monotonic stop boundaries for comparing service order;
those process timestamps are not CUDA completion events.
LMCache MP's Uvicorn lifespan shutdown may re-emit the requested SIGTERM and
return `-15`. Only this explicit protocol accepts that literal status, after
checking fresh, ordered cache/HTTP shutdown logs, the exact service PID, actual
signal delivery, reap and empty process ownership. Forced cleanup, premature
exit, error-level shutdown diagnostics and incomplete logs invalidate the run.
Other services still require zero.
Before running released LMCache comparisons, run the real CPU Uvicorn control
with the installed runtime's Uvicorn version (currently 0.54.0):

```bash
uv run --isolated --no-project --with pytest --with requests --with uvicorn==0.54.0 \
  pytest -q benches/tests/common/test_runtime.py -k 'lmcache_signal_exit or uvicorn_reemits'
```


`launch`, `runtime`, `shutdown`, `workload`, `metrics`, `timeline` and `report` own shared harness behavior. `artifacts` validates external output paths. They are consumed directly by the family entry points; there is no forwarding CLI layer.
