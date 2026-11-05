"""Fixed-capacity vLLM recovery, preemption and client-disconnect gate.

This proves output after ordinary scheduler preemption and cancellation after
the first generated token. It does not inject cancellation during physical DMA.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest
import requests

from tests.support.vllm_helpers import CacheManager, VLLMServer

pytestmark = [pytest.mark.e2e, pytest.mark.stress, pytest.mark.gpu]

PROMPT_TOKENS = 512
OUTPUT_TOKENS = 256
GPU_BLOCKS = 32
BLOCK_SIZE = 64
CONCURRENCY = 4


def _completion(port: int, model: str, prompt: list[int], *, cancel: bool = False) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512 if cancel else OUTPUT_TOKENS,
        "temperature": 0.0,
        "seed": 42,
        "ignore_eos": True,
        "stream": cancel,
        "logprobs": 0,
        "return_tokens_as_token_ids": True,
    }
    with requests.post(
        f"http://localhost:{port}/v1/completions",
        json=payload,
        stream=cancel,
        timeout=(10, 180),
    ) as response:
        response.raise_for_status()
        if cancel:
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event["choices"] and event["choices"][0]["text"]:
                    return event
            raise AssertionError("Streaming request ended before its first generated text")
        result = response.json()
    assert result["usage"]["completion_tokens"] == OUTPUT_TOKENS, result
    assert result["choices"][0]["finish_reason"] == "length", result
    assert len(result["choices"][0]["logprobs"]["tokens"]) == OUTPUT_TOKENS, result
    return result


def _same_output(actual: dict, expected: dict) -> None:
    actual_choice, expected_choice = actual["choices"][0], expected["choices"][0]
    assert actual_choice["text"] == expected_choice["text"]
    assert actual_choice["logprobs"]["tokens"] == expected_choice["logprobs"]["tokens"]


def _abort_count(port: int) -> float:
    response = requests.get(f"http://localhost:{port}/metrics", timeout=10)
    response.raise_for_status()
    rows = [
        line
        for line in response.text.splitlines()
        if line.startswith("vllm:request_success_total{") and 'finished_reason="abort"' in line
    ]
    assert rows, "Official vLLM abort counter is absent"
    return sum(float(line.rsplit(" ", 1)[1]) for line in rows)


def test_warm_hit_pressure_on_single_gpu_profile(
    model: str, base_port: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Require actual preemption, an acknowledged abort and exact reused-page output."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    from transformers import AutoTokenizer

    from benches.common.metrics import DRAIN_GAUGES, metrics, validate_ownership

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = [
        tokenizer.encode(
            f"Record {topic}: "
            + (
                f"The {topic} study checks memory ownership and exact cache recovery. "
                "Explain how a request preserves data while independent work progresses. "
            )
            * 80,
            add_special_tokens=False,
        )[:PROMPT_TOKENS]
        for topic in ("astronomy", "botany", "chemistry", "geology")
    ]
    assert all(len(prompt) == PROMPT_TOKENS for prompt in prompts)
    assert len({tuple(prompt[:BLOCK_SIZE]) for prompt in prompts}) == CONCURRENCY
    (tmp_path / "prompts.json").write_text(json.dumps(prompts) + "\n")
    report = []

    def record(stage: str, **values) -> None:
        report.append({"stage": stage, **values})
        (tmp_path / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    def server(label: str, manager: CacheManager | None = None) -> VLLMServer:
        return VLLMServer(
            model,
            base_port,
            use_orbitkv=manager is not None,
            orbitkv_port=manager.cache_port if manager else None,
            log_file=tmp_path / f"{label}.log",
            max_model_len=1024,
            gpu_memory_utilization=0.3,
            prefix_caching=True,
            extra_args=[
                "--enforce-eager",
                "--max-num-seqs",
                str(CONCURRENCY),
                "--max-num-batched-tokens",
                "1024",
                "--block-size",
                str(BLOCK_SIZE),
                "--num-gpu-blocks-override",
                str(GPU_BLOCKS),
            ],
        )

    engine_url = f"http://localhost:{base_port}"
    references = []
    with server("native"):
        for index, prompt in enumerate(prompts):
            reference = _completion(base_port, model, prompt)
            references.append(reference)
            record("native_cold", prompt=index, response=reference)
            before = metrics(engine_url)["vllm:prefix_cache_hits_total"]
            warm = _completion(base_port, model, prompt)
            record("native_hbm", prompt=index, response=warm)
            _same_output(warm, reference)
            assert metrics(engine_url)["vllm:prefix_cache_hits_total"] > before

    with CacheManager(
        log_file=tmp_path / "manager.log",
        pool_size="1gb",
        devices="0",
        extra_args=("--query-budget", "512mb"),
    ) as manager:
        manager_url = f"http://localhost:{manager.metrics_port}"

        def settled(stage: str) -> tuple[dict, dict]:
            deadline = time.monotonic() + 30
            while True:
                engine, cache = metrics(engine_url), metrics(manager_url)
                validate_ownership(cache, drained=False)
                if (
                    engine["vllm:num_requests_running"] == 0
                    and engine["vllm:num_requests_waiting"] == 0
                    and engine["vllm:orbitkv_pending_save_requests"] == 0
                    and engine["vllm:orbitkv_pending_prefetches"] == 0
                    and all(cache.get(name, 0) == 0 for name in DRAIN_GAUGES)
                ):
                    validate_ownership(cache, drained=True)
                    record(stage, engine=engine, manager=cache)
                    return engine, cache
                assert time.monotonic() < deadline, (engine, cache)
                time.sleep(0.1)

        with server("populate", manager):
            for index, prompt in enumerate(prompts):
                response = _completion(base_port, model, prompt)
                record("orbitkv_cold", prompt=index, response=response)
                _same_output(response, references[index])
                _, before = settled("before_hbm")
                hits_before = metrics(engine_url)["vllm:prefix_cache_hits_total"]
                warm = _completion(base_port, model, prompt)
                record("orbitkv_hbm", prompt=index, response=warm)
                _same_output(warm, references[index])
                _, after = settled("after_hbm")
                assert after["orbitkv_load_bytes_total"] == before["orbitkv_load_bytes_total"]
                assert metrics(engine_url)["vllm:prefix_cache_hits_total"] > hits_before

        # A fresh engine has no HBM pages; only the Manager survives this restart.
        with server("pressure", manager):
            before_engine, before_cache = settled("before_pressure")
            aborted_before = _abort_count(base_port)
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                futures = {
                    pool.submit(_completion, base_port, model, prompts[index % CONCURRENCY]): index
                    for index in range(3 * CONCURRENCY)
                }
                cancelled = _completion(base_port, model, prompts[0], cancel=True)
                record("client_disconnect_after_first_token", response=cancelled)
                assert any(not future.done() for future in futures), "No concurrent pressure"
                for future in as_completed(futures, timeout=300):
                    index = futures[future]
                    response = future.result()
                    record("pressure", request=index, response=response)
                    _same_output(response, references[index % CONCURRENCY])

            after_engine, after_cache = settled("after_pressure")
            assert _abort_count(base_port) > aborted_before, "Server did not observe the disconnect"
            assert (
                after_engine["vllm:num_preemptions_total"]
                > before_engine["vllm:num_preemptions_total"]
            ), "The fixed-capacity workload did not actually preempt a request"
            assert (
                after_cache["orbitkv_load_bytes_total"] > before_cache["orbitkv_load_bytes_total"]
            )
            for index, prompt in enumerate(prompts):
                response = _completion(base_port, model, prompt)
                record("after_reuse", prompt=index, response=response)
                _same_output(response, references[index])
            settled("final_drain")
