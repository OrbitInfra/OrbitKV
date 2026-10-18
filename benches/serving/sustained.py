"""Bounded closed-loop traffic mixing a reusable working set with cold requests."""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from benches.common.metrics import cached_tokens, codec_summary, measure, percentile, sampled_peak
from benches.common.workload import generate


def run_window(args, base_url, manager_url, vocabulary, prefixes, concurrency, emit):
    samples = []
    with measure(
        base_url,
        manager_url,
        args.settle_seconds,
        backend=args.backend,
        observation_mode=args.observation_mode,
    ) as window:
        started = time.perf_counter()
        deadline = started + args.duration_seconds

        def request(tokens, identity):
            request_started = time.perf_counter() - started
            result = generate(base_url, args.engine, str(args.model), tokens, args.output_tokens)
            result.update(
                identity,
                started_seconds=request_started,
                finished_seconds=time.perf_counter() - started,
                cached_tokens=cached_tokens(args.engine, result),
            )
            prefix = identity["prefix_index"]
            result["matches_reference_output"] = (
                result["text"] == prefixes[prefix]["text"] if prefix is not None else None
            )
            return result

        pending = set()
        submitted = 0
        peak = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while True:
                while len(pending) < concurrency and submitted < args.max_requests:
                    rng = random.Random(f"{args.seed}:{concurrency}:{submitted}")
                    prefix = (
                        rng.randrange(len(prefixes)) if rng.random() < args.reuse_ratio else None
                    )
                    if prefix is None:
                        tokens = [rng.choice(vocabulary) for _ in range(rng.choice(args.lengths))]
                    else:
                        tokens = prefixes[prefix]["tokens"]
                    now = time.perf_counter()
                    if now >= deadline:
                        break
                    identity = {
                        "concurrency": concurrency,
                        "index": submitted,
                        "kind": "reuse" if prefix is not None else "cold",
                        "prefix_index": prefix,
                        "length": len(tokens),
                        "submitted_seconds": now - started,
                    }
                    pending.add(executor.submit(request, tokens, identity))
                    submitted += 1
                    peak = max(peak, len(pending))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    samples.append(result)
                    emit(result)
        window.update(
            concurrency=concurrency,
            submitted_requests=submitted,
            peak_client_inflight=peak,
            stop_reason="request_limit" if submitted == args.max_requests else "duration",
            working_set_tokens=sum(len(prefix["tokens"]) for prefix in prefixes),
        )
    return samples, window


def run_workload(args, base_url: str, manager_url: str | None) -> tuple[list, list]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    vocabulary = tokenizer.encode(
        "The river flows past green trees and quiet houses. A researcher measures memory "
        "transfer latency and checks that repeated requests produce accurate results. ",
        add_special_tokens=False,
    )
    samples, windows = [], []
    with (
        (args.output / "samples.jsonl").open("w") as sample_file,
        (args.output / "windows.jsonl").open("w") as window_file,
        (args.output / "prefixes.jsonl").open("w") as prefix_file,
    ):

        def emit(sample):
            sample_file.write(json.dumps(sample, allow_nan=False) + "\n")
            sample_file.flush()

        for concurrency in args.concurrencies:
            rng = random.Random(f"{args.seed}:{concurrency}:prepare")
            for _ in range(3):
                generate(
                    base_url,
                    args.engine,
                    str(args.model),
                    [rng.choice(vocabulary) for _ in range(256)],
                    args.output_tokens,
                )
            prefixes = []
            for index in range(args.working_set):
                tokens = [
                    rng.choice(vocabulary) for _ in range(args.lengths[index % len(args.lengths)])
                ]
                reference = generate(
                    base_url, args.engine, str(args.model), tokens, args.output_tokens
                )
                prefixes.append({"tokens": tokens, "text": reference["text"]})
                prefix_file.write(
                    json.dumps(
                        {
                            "concurrency": concurrency,
                            "index": index,
                            "tokens": tokens,
                            "reference": reference,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                prefix_file.flush()
            time.sleep(args.settle_seconds)
            rows, window = run_window(
                args, base_url, manager_url, vocabulary, prefixes, concurrency, emit
            )
            samples.extend(rows)
            windows.append(window)
            window_file.write(json.dumps(window, allow_nan=False) + "\n")
            window_file.flush()
            print(
                f"{args.engine}/{args.backend} concurrency={concurrency}: "
                f"{len(rows)} requests in {window['wall_seconds']:.3f}s "
                f"({window['stop_reason']})",
                flush=True,
            )
    return samples, windows


def validate(args: dict, samples: list[dict], windows: list[dict]) -> None:
    expected = args["concurrencies"]
    if len(windows) != len(expected) or {w["concurrency"] for w in windows} != set(expected):
        raise ValueError("Incomplete or duplicated sustained windows")
    if any(s["concurrency"] not in expected for s in samples):
        raise ValueError("Unexpected sustained concurrency")
    for window in windows:
        concurrency = window["concurrency"]
        rows = [s for s in samples if s["concurrency"] == concurrency]
        count = window["submitted_requests"]
        if (
            not 0 < count <= args["max_requests"]
            or len(rows) != count
            or {s["index"] for s in rows} != set(range(count))
        ):
            raise ValueError("Incomplete or duplicated sustained requests")
        if not 0 < window["peak_client_inflight"] <= concurrency:
            raise ValueError("Sustained client exceeded concurrency")
        for field in ("wall_seconds", "drain_seconds"):
            value = window[field]
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid sustained {field}")
        if window["stop_reason"] == "duration":
            if window["wall_seconds"] < args["duration_seconds"]:
                raise ValueError("Sustained window ended before its duration")
        elif window["stop_reason"] != "request_limit" or count != args["max_requests"]:
            raise ValueError("Invalid sustained stop reason")
        if window["wall_seconds"] <= 0:
            raise ValueError("Invalid sustained wall time")
        if window["manager_after"].get("orbitkv_query_reserved_bytes", 0) != 0:
            raise ValueError("Sustained query reservations did not drain")
        query_limit = args.get("query_budget_gib")
        observed = (
            window.get("manager_before", {}),
            window["manager_after"],
            window["sampled_peak_bytes"] or {},
        )
        protected_peak = max(row.get("orbitkv_cache_protected_bytes", 0) for row in observed)
        protected_limit = (
            args.get("host_gib", 0) * 1024**3 * args.get("cache_protected_percent", 0) // 100
        )
        if protected_peak > protected_limit:
            raise ValueError("Protected cache residency exceeded its byte limit")
        if (
            query_limit is not None
            and max(
                max(
                    row.get("orbitkv_query_reserved_bytes", 0),
                    row.get("orbitkv_query_reserved_peak_bytes", 0),
                )
                for row in observed
            )
            > query_limit * 1024**3
        ):
            raise ValueError("Query reservations exceeded their total byte budget")
        if (
            query_limit is not None
            and max(row.get("orbitkv_query_speculative_reserved_bytes", 0) for row in observed)
            > query_limit * 1024**3 / 4
        ):
            raise ValueError("Speculative reservations exceeded their quarter-budget limit")
        for row in rows:
            for field in (
                "ttft_ms",
                "e2e_ms",
                "submitted_seconds",
                "started_seconds",
                "finished_seconds",
            ):
                if not math.isfinite(row[field]) or row[field] < 0:
                    raise ValueError(f"Invalid sustained {field}")
            if not (
                row["ttft_ms"] <= row["e2e_ms"]
                and row["submitted_seconds"] < args["duration_seconds"]
                and row["submitted_seconds"] <= row["started_seconds"] <= row["finished_seconds"]
                and row["finished_seconds"] <= window["wall_seconds"]
            ):
                raise ValueError("Invalid sustained request timing")
            prefix = row["prefix_index"]
            if row["kind"] == "reuse":
                if (
                    not isinstance(prefix, int)
                    or not 0 <= prefix < args["working_set"]
                    or not isinstance(row["matches_reference_output"], bool)
                ):
                    raise ValueError("Invalid sustained reuse evidence")
            elif (
                row["kind"] != "cold"
                or prefix is not None
                or row["matches_reference_output"] is not None
            ):
                raise ValueError("Invalid sustained cold request")


def summarize(samples: list[dict], windows: list[dict]) -> list[dict]:
    result = []
    for window in windows:
        rows = [s for s in samples if s["concurrency"] == window["concurrency"]]
        ttft = [s["ttft_ms"] for s in rows]
        reused = [s for s in rows if s["kind"] == "reuse"]
        cold = [s for s in rows if s["kind"] == "cold"]
        counters = window["manager_delta"]
        result.append(
            {
                "concurrency": window["concurrency"],
                "observation_mode": window.get("observation_mode"),
                "n": len(rows),
                "reuse_requests": len(reused),
                "cold_requests": len(rows) - len(reused),
                "stop_reason": window["stop_reason"],
                "wall_seconds": window["wall_seconds"],
                "drain_seconds": window["drain_seconds"],
                "working_set_tokens": window["working_set_tokens"],
                "ttft_p50_ms": statistics.median(ttft),
                "ttft_p95_ms": percentile(ttft, 0.95),
                "ttft_p99_ms": percentile(ttft, 0.99),
                "reuse_ttft_p95_ms": percentile([s["ttft_ms"] for s in reused], 0.95)
                if reused
                else None,
                "cold_ttft_p95_ms": percentile([s["ttft_ms"] for s in cold], 0.95)
                if cold
                else None,
                "requests_per_second": len(rows) / window["wall_seconds"],
                "output_tokens_per_second": sum(s["usage"]["completion_tokens"] for s in rows)
                / window["wall_seconds"],
                "throughput_scope": "closed-loop window including admitted request completion; excludes preparation and cache drain",
                "e2e_p50_ms": statistics.median(s["e2e_ms"] for s in rows),
                "decode_ms_per_token_p50": statistics.median(
                    (s["e2e_ms"] - s["ttft_ms"]) / max(1, s["usage"]["completion_tokens"] - 1)
                    for s in rows
                ),
                "output_mismatches": sum(not s["matches_reference_output"] for s in reused),
                "cache_sources": {
                    "cached_tier_unknown": sum(s["cached_tokens"] > 0 for s in rows),
                    "miss": sum(s["cached_tokens"] == 0 for s in rows),
                },
                "query_reserved_peak_bytes_since_manager_start": window["manager_after"].get(
                    "orbitkv_query_reserved_peak_bytes"
                ),
                "query_instance_reserved_peak_bytes_since_manager_start": window[
                    "manager_after"
                ].get("orbitkv_query_instance_reserved_peak_bytes"),
                **{
                    f"sampled_peak_{name}_bytes": sampled_peak([window], metric)
                    for name, metric in (
                        ("pool", "orbitkv_pool_used_bytes"),
                        ("protected", "orbitkv_cache_protected_bytes"),
                        ("query", "orbitkv_query_reserved_bytes"),
                        ("warmup", "orbitkv_query_reserved_bytes_warming"),
                        ("speculative", "orbitkv_query_speculative_reserved_bytes"),
                        ("warmup_pending", "orbitkv_warmup_pending_bytes"),
                    )
                },
                "warmup_pending_bytes_before": window.get("manager_before", {}).get(
                    "orbitkv_warmup_pending_bytes", 0
                ),
                "warmup_pending_bytes_after": window["manager_after"].get(
                    "orbitkv_warmup_pending_bytes", 0
                ),
                **{
                    key: counters.get(key, 0)
                    for key in (
                        "orbitkv_ssd_prefetch_bytes_total",
                        "orbitkv_ssd_cufile_read_bytes_total",
                        "orbitkv_ssd_cufile_write_bytes_total",
                        "orbitkv_ssd_cufile_read_seconds_sum",
                        "orbitkv_ssd_cufile_write_seconds_sum",
                        "orbitkv_ssd_write_bytes_total",
                        "orbitkv_load_bytes_total",
                        "orbitkv_save_bytes_total",
                        "orbitkv_query_budget_waits_total",
                        "orbitkv_query_budget_bypasses_total",
                        "orbitkv_query_coalesced_reads_total",
                        "orbitkv_warmup_prepared_bytes_total",
                        "orbitkv_warmup_restored_bytes_total",
                        "orbitkv_warmup_unused_bytes_total",
                        "orbitkv_warmup_foreground_skips_total",
                        "orbitkv_warmup_wait_byte_seconds_total_restored",
                        "orbitkv_warmup_wait_byte_seconds_total_unused",
                        "orbitkv_load_duration_seconds_sum",
                        "orbitkv_load_duration_seconds_count",
                        "orbitkv_ssd_prefetch_duration_seconds_sum",
                        "orbitkv_ssd_prefetch_duration_seconds_count",
                        "orbitkv_save_duration_seconds_sum",
                        "orbitkv_save_duration_seconds_count",
                        "orbitkv_pool_alloc_failures_total",
                        "orbitkv_ssd_prefetch_failures_total",
                        "orbitkv_ssd_write_queue_full_total",
                        "orbitkv_cache_policy_promotions_total",
                        "orbitkv_cache_policy_demotions_total",
                        "orbitkv_ssd_write_admission_skips_total_cold",
                        "orbitkv_ssd_write_admission_skips_total_resident",
                        "orbitkv_ssd_write_admission_skips_total_pending",
                        "orbitkv_ssd_write_admission_skips_total_duplicate",
                    )
                },
                **codec_summary([window]),
            }
        )
    return result
