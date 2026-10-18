"""Closed-loop bursts with shared prefixes, mixed lengths, and batch-level tier evidence."""

from __future__ import annotations

import json
import math
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from benches.common.metrics import cached_tokens, codec_summary, measure, percentile, sampled_peak
from benches.common.workload import evict_host_cache, generate

PATTERNS = ("shared", "mixed")


def phases(ssd: bool) -> tuple[str, ...]:
    return ("cold", "after_pressure", "after_host_eviction") if ssd else ("cold", "after_pressure")


def burst(
    args, base_url: str, manager_url: str | None, prompts: list[list[int]]
) -> tuple[list, dict]:
    barrier = threading.Barrier(len(prompts))

    def request(tokens):
        barrier.wait(timeout=30)
        return generate(base_url, args.engine, str(args.model), tokens, args.output_tokens)

    with (
        measure(
            base_url,
            manager_url,
            args.settle_seconds,
            backend=args.backend,
            observation_mode=args.observation_mode,
        ) as measurement,
        ThreadPoolExecutor(max_workers=len(prompts)) as executor,
    ):
        results = list(executor.map(request, prompts))
    return results, measurement


def run_workload(args, base_url: str, manager_url: str | None) -> tuple[list, list]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    vocabulary = tokenizer.encode(
        "The river flows past green trees and quiet houses. A researcher measures memory "
        "transfer latency and checks that repeated requests produce accurate results. ",
        add_special_tokens=False,
    )
    rng = random.Random(args.seed)

    def prompt(length):
        return [rng.choice(vocabulary) for _ in range(length)]

    for _ in range(3):
        generate(base_url, args.engine, str(args.model), prompt(256), args.output_tokens)
    samples, batches = [], []
    with (
        (args.output / "samples.jsonl").open("w") as sample_file,
        (args.output / "batches.jsonl").open("w") as batch_file,
    ):
        for concurrency in args.concurrencies:
            for pattern in PATTERNS:
                for repeat in range(args.repeats):
                    if pattern == "shared":
                        prompts = [prompt(args.lengths[repeat % len(args.lengths)])] * concurrency
                    else:
                        prompts = [
                            prompt(args.lengths[(repeat + i) % len(args.lengths)])
                            for i in range(concurrency)
                        ]
                    cold = None
                    for phase in phases(bool(args.ssd_gib)):
                        preparation = {}
                        if phase != "cold":
                            for _ in range(2):
                                generate(
                                    base_url,
                                    args.engine,
                                    str(args.model),
                                    prompt(args.gpu_tokens * 3 // 4),
                                    1,
                                )
                        time.sleep(args.settle_seconds)
                        if phase == "after_host_eviction":
                            preparation = evict_host_cache(manager_url)
                        responses, batch = burst(args, base_url, manager_url, prompts)
                        identity = {
                            "concurrency": concurrency,
                            "pattern": pattern,
                            "repeat": repeat,
                            "phase": phase,
                        }
                        if cold is None:
                            cold = [response["text"] for response in responses]
                        for index, (tokens, result) in enumerate(
                            zip(prompts, responses, strict=True)
                        ):
                            result.update(
                                identity,
                                index=index,
                                length=len(tokens),
                                matches_cold_output=result["text"] == cold[index],
                            )
                            result["cached_tokens"] = cached_tokens(args.engine, result)
                            samples.append(result)
                            sample_file.write(json.dumps(result, allow_nan=False) + "\n")
                        sample_file.flush()
                        batch.update(identity, preparation=preparation)
                        batches.append(batch)
                        batch_file.write(json.dumps(batch, allow_nan=False) + "\n")
                        batch_file.flush()
                        print(
                            f"{args.engine} concurrency={concurrency} pattern={pattern} repeat={repeat} phase={phase}: {batch['wall_seconds']:.3f}s",
                            flush=True,
                        )
    return samples, batches


def summarize(samples: list[dict], batches: list[dict]) -> list[dict]:
    result = []
    groups = list(dict.fromkeys((s["concurrency"], s["pattern"], s["phase"]) for s in samples))
    for concurrency, pattern, phase in groups:

        def matches(row, group=(concurrency, pattern, phase)):
            return (row["concurrency"], row["pattern"], row["phase"]) == group

        rows = [s for s in samples if matches(s)]
        measurements = [b for b in batches if matches(b)]
        ttft = [s["ttft_ms"] for s in rows]
        seconds = sum(b["wall_seconds"] for b in measurements)
        manager = {
            key: sum(b["manager_delta"].get(key, 0) for b in measurements)
            for key in (
                "orbitkv_ssd_prefetch_bytes_total",
                "orbitkv_ssd_cufile_read_bytes_total",
                "orbitkv_ssd_cufile_write_bytes_total",
                "orbitkv_ssd_cufile_read_seconds_sum",
                "orbitkv_ssd_cufile_write_seconds_sum",
                "orbitkv_load_bytes_total",
                "orbitkv_query_budget_waits_total",
                "orbitkv_query_budget_bypasses_total",
                "orbitkv_query_coalesced_reads_total",
                "orbitkv_load_duration_seconds_sum",
                "orbitkv_load_duration_seconds_count",
                "orbitkv_ssd_prefetch_duration_seconds_sum",
                "orbitkv_ssd_prefetch_duration_seconds_count",
            )
        }
        result.append(
            {
                "concurrency": concurrency,
                "pattern": pattern,
                "phase": phase,
                "observation_mode": measurements[0].get("observation_mode"),
                "n": len(rows),
                "ttft_p50_ms": statistics.median(ttft),
                "ttft_p95_ms": percentile(ttft, 0.95),
                "ttft_p99_ms": percentile(ttft, 0.99),
                "requests_per_second": len(rows) / seconds,
                "output_tokens_per_second": sum(s["usage"]["completion_tokens"] for s in rows)
                / seconds,
                "wall_seconds": seconds,
                "throughput_scope": "closed-loop bursts; excludes pressure and cache drain",
                "e2e_p50_ms": statistics.median(s["e2e_ms"] for s in rows),
                "decode_ms_per_token_p50": statistics.median(
                    (s["e2e_ms"] - s["ttft_ms"]) / max(1, s["usage"]["completion_tokens"] - 1)
                    for s in rows
                ),
                "output_mismatches": sum(not s["matches_cold_output"] for s in rows),
                "cache_sources": {
                    "cached_tier_unknown": sum(s["cached_tokens"] > 0 for s in rows),
                    "miss": sum(s["cached_tokens"] == 0 for s in rows),
                },
                "sampled_peak_pool_bytes": sampled_peak(measurements, "orbitkv_pool_used_bytes"),
                "sampled_peak_query_bytes": sampled_peak(
                    measurements, "orbitkv_query_reserved_bytes"
                ),
                "sampled_peak_speculative_bytes": sampled_peak(
                    measurements, "orbitkv_query_speculative_reserved_bytes"
                ),
                **manager,
                **codec_summary(measurements),
            }
        )
    return result


def validate(args: dict, samples: list[dict], batches: list[dict]) -> None:
    for sample in samples:
        for field in ("ttft_ms", "e2e_ms"):
            if not math.isfinite(sample[field]) or sample[field] < 0:
                raise ValueError(f"Invalid concurrent {field}")
    for batch in batches:
        if not math.isfinite(batch["wall_seconds"]) or batch["wall_seconds"] <= 0:
            raise ValueError("Invalid concurrent wall time")
    expected_batches = {
        (c, pattern, repeat, phase)
        for c in args["concurrencies"]
        for pattern in PATTERNS
        for repeat in range(args["repeats"])
        for phase in phases(bool(args.get("ssd_gib", 0)))
    }

    def key(row):
        return row["concurrency"], row["pattern"], row["repeat"], row["phase"]

    if len(batches) != len(expected_batches) or {key(b) for b in batches} != expected_batches:
        raise ValueError("Incomplete or duplicated concurrent batches")
    expected_samples = {(*batch, index) for batch in expected_batches for index in range(batch[0])}
    if (
        len(samples) != len(expected_samples)
        or {(*key(s), s["index"]) for s in samples} != expected_samples
    ):
        raise ValueError("Incomplete or duplicated concurrent requests")
