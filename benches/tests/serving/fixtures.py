"""Shared deterministic inputs for this benchmark family."""


def evidence():
    args = {
        "engine": "vllm",
        "backend": "native",
        "observation_mode": "boundary",
        "workload": "sustained",
        "concurrencies": [2],
        "max_requests": 100,
        "duration_seconds": 1,
        "working_set": 1,
    }
    rows = [
        {
            "concurrency": 2,
            "index": i,
            "kind": "reuse",
            "prefix_index": 0,
            "submitted_seconds": 0.99,
            "started_seconds": 0.99,
            "finished_seconds": 1.1,
            "ttft_ms": 10,
            "e2e_ms": 110,
            "usage": {"completion_tokens": 2},
            "cached_tokens": 64,
            "matches_reference_output": i == 0,
        }
        for i in range(2)
    ]
    windows = [
        {
            "concurrency": 2,
            "submitted_requests": 2,
            "peak_client_inflight": 2,
            "stop_reason": "duration",
            "wall_seconds": 1.1,
            "drain_seconds": 0.2,
            "working_set_tokens": 64,
            "manager_after": {},
            "manager_before": {},
            "sampled_peak_bytes": None,
            "observation_mode": "boundary",
            "manager_present": False,
            "manager_backend": None,
            "memory_samples": 0,
            "manager_delta": {"orbitkv_ssd_prefetch_bytes_total": 4096},
        }
    ]
    return args, rows, windows
