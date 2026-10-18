"""Concurrent model-serving faults; requires the explicit test-hooks Manager.

Run each engine with its pinned environment, ORBITKV_FAULT_TESTS=1,
ORBITKV_CACHE_MANAGER_BINARY and --model /path/to/qwen3-8b.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import signal
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

pytestmark = [
    pytest.mark.stress,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("ORBITKV_FAULT_TESTS") != "1", reason="requires test-hooks Manager"
    ),
]


def until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not (result := predicate()):
        assert time.monotonic() < deadline, "serving fault gate timed out"
        time.sleep(0.02)
    return result


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_concurrent_model_recovery_faults(engine, request, tmp_path, monkeypatch):
    pytest.importorskip(engine)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    from benches.common.launch import configure
    from benches.common.metrics import metrics
    from benches.common.runtime import server
    from benches.common.workload import evict_host_cache, generate

    model = Path(request.config.getoption("--model"))
    config = json.loads((model / "config.json").read_text())
    assert config["model_type"] == "qwen3", "this gate qualifies dense Qwen3"
    assert os.environ.get("ORBITKV_CACHE_MANAGER_BINARY"), "select the test-hooks binary explicitly"
    args = Namespace(
        engine=engine,
        backend="orbitkv",
        model=model,
        output=tmp_path,
        host_gib=2,
        gpu_tokens=8192,
        ssd_gib=8,
        query_budget_gib=1,
        queue_warmup="off",
        prepare_requests="on" if os.environ.get("ORBITKV_PREPARE_REQUESTS") == "1" else "off",
        read_batch_mib=32,
        read_timeout_ms=0,
        read_max_batches=0,
        trace_transfers=True,
        orbitkv_transfer_backend=None,
    )
    bpt = 4 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"]
    launch = configure(args, bpt)
    barriers = tmp_path / "faults"
    barriers.mkdir()
    launch.env["ORBITKV_TEST_FAULTS"] = str(barriers)
    if engine == "vllm":
        launch.env["VLLM_BATCH_INVARIANT"] = "1"
    else:
        launch.command.append("--enable-deterministic-inference")
    from transformers import AutoTokenizer

    vocabulary = AutoTokenizer.from_pretrained(model, local_files_only=True).encode(
        "A researcher verifies memory ownership while independent requests continue. "
    )
    rng = random.Random(20260922)
    prompt = [rng.choice(vocabulary) for _ in range(1025)]
    cold = [rng.choice(vocabulary) for _ in range(769)]
    report = []

    def arm(name):
        (barriers / f"{name}.reached").unlink(missing_ok=True)
        (barriers / f"{name}.pause").touch()

    def reached(name):
        until(lambda: (barriers / f"{name}.reached").exists())

    def release(name):
        (barriers / f"{name}.pause").unlink(missing_ok=True)

    def drain():
        def idle():
            stats = metrics(launch.manager_url)
            return (
                stats
                if not any(
                    stats.get(name, 0)
                    for name in (
                        "orbitkv_query_reserved_bytes",
                        "orbitkv_inflight_bytes",
                        "orbitkv_ssd_prefetch_inflight",
                        "orbitkv_ssd_write_queue_pending",
                        "orbitkv_ssd_write_inflight",
                    )
                )
                else None
            )

        return until(idle)

    def infer(tokens):
        return generate(launch.base_url, engine, str(model), tokens, 8)

    manager_stack, engine_stack = contextlib.ExitStack(), contextlib.ExitStack()
    manager_logs, engine_logs = [], []

    def start_manager():
        manager_logs.append(tmp_path / f"manager-{len(manager_logs)}.log")
        return manager_stack.enter_context(
            server(
                launch.manager_command,
                launch.env,
                launch.manager_url,
                manager_logs[-1],
                launch.manager_health_path,
            )
        )

    def start_engine():
        engine_logs.append(tmp_path / f"engine-{len(engine_logs)}.log")
        return engine_stack.enter_context(
            server(launch.command, launch.env, launch.base_url, engine_logs[-1])
        )

    pool = ThreadPoolExecutor(4)
    try:
        manager = start_manager()
        start_engine()
        baseline = infer(prompt)
        cold_reference = infer(cold)
        drain()
        # Engine restart removes HBM residency while retaining Manager copies.
        engine_stack.close()
        start_engine()
        before = metrics(launch.manager_url)
        warm = infer(prompt)
        assert warm["text"] == baseline["text"]
        assert metrics(launch.manager_url)["orbitkv_load_bytes_total"] > before.get(
            "orbitkv_load_bytes_total", 0
        )
        report.append({"case": "engine_restart", "result": warm})

        def evict():
            # Force target pages out of HBM through ordinary engine pressure.
            for _ in range(2):
                infer([rng.choice(vocabulary) for _ in range(6144)])
            drain()
            assert evict_host_cache(launch.manager_url)["cleanup"]["evicted_blocks"] > 0

        evict()
        arm("ssd")
        endpoint = "/v1/completions" if engine == "vllm" else "/generate"
        payload = (
            {
                "model": str(model),
                "prompt": prompt,
                "max_tokens": 8,
                "temperature": 0,
                "stream": True,
            }
            if engine == "vllm"
            else {
                "input_ids": prompt,
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                "stream": True,
            }
        )
        abandoned = requests.post(
            launch.base_url + endpoint, json=payload, stream=True, timeout=(10, 30)
        )
        try:
            abandoned.raise_for_status()
            reached("ssd")
            before_cancel = manager_logs[-1].read_text().count('"stage":"query_cancel"')
            abandoned.close()
            until(
                lambda: manager_logs[-1].read_text().count('"stage":"query_cancel"') > before_cancel
            )
            assert metrics(launch.manager_url)["orbitkv_query_reserved_bytes"] > 0
            independent = pool.submit(infer, [rng.choice(vocabulary) for _ in range(513)])
            assert independent.result(timeout=30)["text"]
        finally:
            abandoned.close()
            release("ssd")
        report.append({"case": "ssd_cancel", "drained": drain()})

        evict()
        arm("notification")
        before = metrics(launch.manager_url)
        restored = pool.submit(infer, prompt)
        other = pool.submit(infer, cold)
        try:
            assert restored.result(timeout=60)["text"] == baseline["text"]
            assert other.result(timeout=60)["text"] == cold_reference["text"]
            reached("notification")
            assert metrics(launch.manager_url)["orbitkv_load_bytes_total"] > before.get(
                "orbitkv_load_bytes_total", 0
            )
        finally:
            release("notification")
        report.append({"case": "lost_notification", "drained": drain()})

        # Restart while a real model request owns a pending GPU restore.
        evict()
        arm("restore")
        pending = pool.submit(infer, prompt)
        reached("restore")
        os.kill(manager.pid, signal.SIGKILL)
        engine_stack.close()
        with pytest.raises((requests.RequestException, RuntimeError, ValueError)):
            pending.result(timeout=30)
        release("restore")
        manager_stack.close()
        start_manager()
        start_engine()
        assert infer(prompt)["text"] == baseline["text"]
        report.append({"case": "manager_restart", "drained": drain()})
    finally:
        for pause in barriers.glob("*.pause"):
            pause.unlink(missing_ok=True)
        engine_stack.close()
        manager_stack.close()
        pool.shutdown(wait=True)
        (tmp_path / "fault-results.json").write_text(json.dumps(report, indent=2) + "\n")
