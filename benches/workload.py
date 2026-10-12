"""Token-exact streaming requests and serial cold/resident/pressure phases."""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
from argparse import Namespace

import requests

from .metrics import cache_source, measure, metrics, workload_phases
from .sglang_profile import sglang_profile


def generate(url: str, engine: str, model: str, tokens: list[int], output_len: int) -> dict:
    prompt_sha256 = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
    if engine == "vllm":
        endpoint = "/v1/completions"
        payload = {
            "model": model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        endpoint = "/generate"
        payload = {
            "input_ids": tokens,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "stream": True,
        }
    started = time.perf_counter()
    first = None
    text = ""
    usage = {}
    request_id = None
    with requests.post(url + endpoint, json=payload, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if engine == "vllm":
                request_id = event.get("id", request_id)
                chunk = "".join(choice.get("text", "") for choice in event.get("choices", []))
                text += chunk
                usage = event.get("usage") or usage
            else:
                chunk = event.get("text", "")
                text = chunk
                usage = event.get("meta_info", usage)
                request_id = usage.get("id", request_id)
            if first is None and chunk:
                first = time.perf_counter()
    ended = time.perf_counter()
    if first is None:
        raise RuntimeError("No generated text in streaming response")
    if usage.get("prompt_tokens") != len(tokens):
        raise RuntimeError(f"Input length changed: expected {len(tokens)}, got {usage}")
    if usage.get("completion_tokens") != output_len:
        raise RuntimeError(f"Output length changed: expected {output_len}, got {usage}")
    return {
        "request_id": request_id,
        "ttft_ms": (first - started) * 1000,
        "e2e_ms": (ended - started) * 1000,
        "text": text,
        "usage": usage,
        "prompt_sha256": prompt_sha256,
    }


def evict_host_cache(manager_url: str) -> dict:
    deadline = time.monotonic() + 60
    previous = None
    quiet = 0
    while time.monotonic() < deadline:
        observed = metrics(manager_url)
        stamp = tuple(
            observed.get(name, 0)
            for name in ("orbitkv_save_bytes_total", "orbitkv_ssd_write_bytes_total")
        )
        busy = any(
            observed.get(name, 0) != 0
            for name in (
                "orbitkv_ssd_write_queue_pending",
                "orbitkv_ssd_write_inflight",
                "orbitkv_ssd_prefetch_inflight",
                "orbitkv_inflight_bytes",
                "orbitkv_ssd_cufile_inflight_batches",
                "orbitkv_storage_codec_reserved_bytes",
            )
        )
        quiet = quiet + 1 if not busy and stamp == previous else 0
        if quiet >= 3:
            break
        previous = stamp
        time.sleep(0.1)
    else:
        raise TimeoutError("Cache Manager did not become idle before DRAM eviction")
    response = requests.post(f"{manager_url}/cache/memory/cleanup", timeout=30)
    response.raise_for_status()
    result = response.json()
    if result["still_referenced_blocks"] or not result["evicted_blocks"]:
        raise RuntimeError(f"Could not establish an empty manager DRAM cache: {result}")
    return {"cleanup": result, "manager_before_cleanup": observed}


def run_workload(args: Namespace, base_url: str, manager_url: str | None) -> list[dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    vocabulary = tokenizer.encode(
        "The river flows past green trees and quiet houses. A researcher measures memory "
        "transfer latency and checks that repeated requests produce accurate results. ",
        add_special_tokens=False,
    )
    rng = random.Random(args.seed)

    def prompt(length: int) -> list[int]:
        # Distinct first pages prevent unintended sharing across independent prefixes.
        return [rng.choice(vocabulary) for _ in range(length)]

    samples = []
    pressure_tokens = args.gpu_tokens * 3 // 4
    for _ in range(3):
        generate(base_url, args.engine, str(args.model), prompt(256), args.output_tokens)
    with (args.output / "samples.jsonl").open("w") as output:
        for length in args.lengths:
            generate(
                base_url,
                args.engine,
                str(args.model),
                prompt(length),
                args.output_tokens,
            )
            for repeat in range(args.repeats):
                tokens = prompt(length)
                texts = []
                for phase in workload_phases(bool(args.ssd_gib)):
                    preparation = {}
                    if phase in ("after_pressure", "after_host_eviction"):
                        for _ in range(2):
                            generate(
                                base_url,
                                args.engine,
                                str(args.model),
                                prompt(pressure_tokens),
                                1,
                            )
                    time.sleep(args.settle_seconds)
                    if phase == "after_host_eviction":
                        preparation = evict_host_cache(manager_url)
                    with measure(
                        base_url,
                        manager_url,
                        args.settle_seconds,
                        backend=args.backend,
                        observation_mode=args.observation_mode,
                    ) as measurement:
                        profiling = {"mode": "off"}
                        profile = (
                            sglang_profile(
                                base_url,
                                args.output / "profiles" / f"{length}-{repeat}-{phase}",
                            )
                            if args.sglang_profile and phase == "after_pressure"
                            else contextlib.nullcontext(profiling)
                        )
                        with profile as profiling:
                            result = generate(
                                base_url,
                                args.engine,
                                str(args.model),
                                tokens,
                                args.output_tokens,
                            )
                    result.update(
                        length=length,
                        repeat=repeat,
                        phase=phase,
                        preparation=preparation,
                        profiling=profiling,
                        **measurement,
                    )
                    texts.append(result["text"])
                    result["cache_source"] = cache_source(args.engine, result)
                    result["matches_cold_output"] = result["text"] == texts[0]
                    samples.append(result)
                    output.write(json.dumps(result, allow_nan=False) + "\n")
                    output.flush()
                    print(
                        f"{args.engine}/{args.backend} len={length} repeat={repeat} {phase}: TTFT={result['ttft_ms']:.2f}ms usage={result['usage']}",
                        flush=True,
                    )
    return samples
