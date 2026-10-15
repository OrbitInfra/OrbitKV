"""Measure admitted-query polling through the real Cache Manager channel.

Hold the instance byte budget with a ready lease so measured operations stay
Loading without doing storage I/O. This measures client/control overhead, not
serving latency, GPU transfer throughput, or cache policy effectiveness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import sysconfig
import time

from .artifacts import external_path
from .runtime import ROOT, free_port, server


def main() -> None:
    import torch

    from orbitkv import BlockHashes, CacheManagerClient, QueryLoading, QueryReady

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=external_path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    port, http = free_port(), free_port()
    env = dict(os.environ, PYO3_PYTHON=sys.executable, PYTHONHOME=sys.base_prefix)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "python"), sysconfig.get_path("purelib")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [sysconfig.get_config_var("LIBDIR"), env.get("LD_LIBRARY_PATH", "")]
    )
    command = [
        str(ROOT / "target/release/orbitkv-cache-manager"),
        "--addr",
        f"127.0.0.1:{port}",
        "--http-addr",
        f"127.0.0.1:{http}",
        "--pool-size",
        "256mb",
        "--query-budget",
        str(64 * 1024**2),
        "--query-instance-budget",
        str(64 * 1024**2),
    ]
    samples = []
    with server(command, env, f"http://127.0.0.1:{http}", args.output / "manager.log"):
        client = CacheManagerClient(f"/tmp/orbitkv-{port}.sock")
        pages = torch.zeros((2, 1024, 16, 8, 128), dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        hashes = [hashlib.sha256(str(index).encode()).digest() for index in range(1024)]
        try:
            registration = client.register_context_batch(
                "poll-bench",
                "poll-bench",
                0,
                0,
                1,
                1,
                0,
                ["layer"],
                [1024],
                [32768],
                [32 * 1024**2],
                [2],
                "direct",
                False,
                tensors=[pages],
            )
            client.save(registration, [("layer", list(range(1024)), hashes)])
            demand = BlockHashes(hashes)
            deadline = time.monotonic() + 10
            while True:
                held = client.query_prefetch("poll-bench", demand, "budget-owner")
                if isinstance(held, QueryReady) and held.num_hit_blocks == len(hashes):
                    break
                if isinstance(held, QueryReady) and held.lease:
                    client.release(held.lease)
                if time.monotonic() >= deadline:
                    raise TimeoutError("could not retain the complete instance budget")
                time.sleep(0.001)
            try:
                for repeat in range(args.repeats):
                    for count in (64, 256, 1024):
                        query = demand[:count]
                        request = f"pending-{repeat}-{count}"
                        first = client.query_prefetch("poll-bench", query, request)
                        assert isinstance(first, QueryLoading) and first.admitted
                        timings = []
                        cpu_started = time.thread_time_ns()
                        for _ in range(args.iterations):
                            started = time.perf_counter_ns()
                            result = client.query_prefetch("poll-bench", query, request)
                            timings.append((time.perf_counter_ns() - started) / 1000)
                            assert isinstance(result, QueryLoading) and result.admitted
                        cpu_us = (time.thread_time_ns() - cpu_started) / args.iterations / 1000
                        client.cancel_query("poll-bench", request)
                        samples.append(
                            {
                                "repeat": repeat,
                                "hashes": count,
                                "calls": args.iterations,
                                "p50_us": statistics.median(timings),
                                "p95_us": sorted(timings)[int(len(timings) * 0.95)],
                                "mean_us": statistics.mean(timings),
                                "thread_cpu_us": cpu_us,
                            }
                        )
            finally:
                client.release(held.lease)
        finally:
            client.unregister_context("poll-bench")
            client.close()
            registration = None
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "label": args.label,
                "python": sys.version,
                "gpu": torch.cuda.get_device_name(),
                "command": command,
                "samples": samples,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
