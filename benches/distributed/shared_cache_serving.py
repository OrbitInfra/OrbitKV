"""Check first and repeated remote serving restores on dedicated matching replicas.

Services are started separately. Timers cover the existing streaming request;
idle HBM resets, DRAM eviction, metadata fences and resource checks stay outside
that timer and are recorded separately. This is a descriptive correctness gate,
not a matched backend performance qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import requests

from benches.common.artifacts import external_path
from benches.common.workload import generate
from benches.distributed.shared_cache import (
    IDLE_METRICS,
    drain,
    synchronize,
    verify_restore,
    verify_source_ssd,
)


def reset_hbm(url: str, engine: str) -> dict:
    path = "/reset_prefix_cache" if engine == "vllm" else "/flush_cache?timeout=30"
    started = time.monotonic_ns()
    deadline = time.monotonic() + 30
    attempts = 0
    while True:
        response = requests.post(url + path, timeout=40)
        response.raise_for_status()
        attempts += 1
        if engine != "vllm" or response.json()["success"]:
            return {
                "path": path,
                "attempts": attempts,
                "response": response.text,
                "elapsed_ms": (time.monotonic_ns() - started) / 1e6,
            }
        if time.monotonic() >= deadline:
            raise TimeoutError("Idle HBM-only reset did not succeed")
        time.sleep(0.1)


def clear_dram(url: str) -> dict:
    (before,) = drain(url)
    started = time.monotonic_ns()
    response = requests.post(url + "/cache/memory/cleanup", timeout=30)
    response.raise_for_status()
    cleanup = response.json()
    if cleanup["still_referenced_blocks"]:
        raise AssertionError(f"DRAM remains referenced: {cleanup}")
    (after,) = drain(url)
    if "orbitkv_cache_resident_bytes" not in after or after["orbitkv_cache_resident_bytes"] != 0:
        raise AssertionError("DRAM cleanup did not establish zero resident bytes")
    return {
        "cleanup": cleanup,
        "before": before,
        "after": after,
        "elapsed_ms": (time.monotonic_ns() - started) / 1e6,
    }


def restore_evidence(
    before: list[dict], after: list[dict], expected: dict, actual: dict, payload_bytes: int
) -> dict:
    result = verify_restore(before[1], after[1], expected["text"], actual)
    if result["remote_bytes"] != payload_bytes or result["h2d_bytes"] != payload_bytes:
        raise AssertionError(f"Expected {payload_bytes} remote/H2D bytes: {result}")
    if (
        actual["usage"]["prompt_tokens"] != expected["usage"]["prompt_tokens"]
        or actual["usage"]["completion_tokens"] != expected["usage"]["completion_tokens"]
    ):
        raise AssertionError("Remote response token counts differ from the cold control")
    required = (
        ("orbitkv_transfer_lock_active", "orbitkv_transfer_reserved_bytes"),
        ("orbitkv_query_reserved_bytes", "orbitkv_transfer_completion_outstanding"),
    )
    for snapshot, names in zip(after, required, strict=True):
        for name in names:
            if name not in snapshot or snapshot[name] != 0:
                raise AssertionError(f"Missing or nonzero ownership gauge: {name}")
    return {
        **result,
        "manager_before": before,
        "manager_after": after,
        "source_response": expected,
        "consumer_response": actual,
        "ownership": [
            {
                "observed_zero": {
                    name: snapshot[name] for name in IDLE_METRICS if name in snapshot
                },
                "unexported": [name for name in IDLE_METRICS if name not in snapshot],
            }
            for snapshot in after
        ],
    }


def profile(args: argparse.Namespace, prompts: list[list[int]]) -> dict:
    if args.source_url.rstrip("/") == args.target_url.rstrip("/") or args.source_manager.rstrip(
        "/"
    ) == args.target_manager.rstrip("/"):
        raise ValueError("Independent source and consumer services are required")
    warmup_repeats = getattr(args, "warmup_repeats", 0)
    if (
        args.bytes_per_token <= 0
        or args.block_tokens <= 0
        or not 1 <= args.remote_repeats <= 10000
        or not 0 <= warmup_repeats <= 10000 - args.remote_repeats
    ):
        raise ValueError("Invalid payload geometry or repeat count")
    if args.output.exists():
        raise FileExistsError(f"Preserve the existing profile: {args.output}")
    result = {
        "engine": args.engine,
        "model": args.model,
        "source_medium": args.source_medium,
        "bytes_per_token": args.bytes_per_token,
        "block_tokens": args.block_tokens,
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(prompts, separators=(",", ":")).encode()
        ).hexdigest(),
        "samples": [],
        "status": "RUNNING",
        "performance_qualified": False,
        "warmup_repeats_per_prompt": warmup_repeats,
        "measured_repeats_per_prompt": args.remote_repeats,
        "timing_scope": "Client TTFT to first nonempty text; preparation and drain excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for index, prompt in enumerate(prompts):
            payload_bytes = (
                len(prompt) // args.block_tokens * args.block_tokens * args.bytes_per_token
            )
            if payload_bytes <= 0:
                raise ValueError("Prompt must contain at least one complete block")
            source_before, _ = drain(args.source_manager, args.target_manager)
            expected = generate(
                args.source_url, args.engine, args.model, prompt, args.output_tokens
            )
            source_fence = synchronize(args.source_manager, args.target_manager)
            source_after, _ = drain(args.source_manager, args.target_manager)
            if (
                source_after.get("orbitkv_save_bytes_total", 0)
                - source_before.get("orbitkv_save_bytes_total", 0)
                < payload_bytes
            ):
                raise AssertionError("Source did not publish the complete fresh prefix")
            if source_after.get("orbitkv_load_bytes_total", 0) != source_before.get(
                "orbitkv_load_bytes_total", 0
            ):
                raise AssertionError("Source cold control restored another replica")
            if (
                args.source_medium == "ssd"
                and source_after.get("orbitkv_ssd_write_bytes_total", 0)
                - source_before.get("orbitkv_ssd_write_bytes_total", 0)
                < payload_bytes
            ):
                raise AssertionError("Source did not commit the complete prefix to SSD")
            for repeat in range(warmup_repeats + args.remote_repeats):
                preparation_started = time.monotonic_ns()
                preparation = {"source_fence": source_fence}
                if repeat:
                    preparation["hbm_reset"] = reset_hbm(args.target_url, args.engine)
                    preparation["target_dram"] = clear_dram(args.target_manager)
                    preparation["target_fence"] = synchronize(
                        args.target_manager, args.source_manager
                    )
                if args.source_medium == "ssd":
                    preparation["source_dram"] = clear_dram(args.source_manager)
                    preparation["source_fence"] = synchronize(
                        args.source_manager, args.target_manager
                    )
                before = drain(args.source_manager, args.target_manager)
                preparation["elapsed_ms"] = (time.monotonic_ns() - preparation_started) / 1e6
                row = {
                    "prompt": index,
                    "repeat": repeat,
                    "phase": "first_peer_read" if index == 0 and repeat == 0 else "warm_peer_read",
                    "sample_kind": "warmup" if repeat < warmup_repeats else "measured",
                    "payload_bytes": payload_bytes,
                    "preparation": preparation,
                    "manager_before": before,
                    "source_response": expected,
                    "status": "REQUEST_STARTED",
                }
                result["samples"].append(row)
                save()
                row["request_start_mono_ns"] = time.monotonic_ns()
                row["request_start_wall_ns"] = time.time_ns()
                actual = generate(
                    args.target_url, args.engine, args.model, prompt, args.output_tokens
                )
                row["request_end_mono_ns"] = time.monotonic_ns()
                after = drain(args.source_manager, args.target_manager)
                row.update(consumer_response=actual, manager_after=after, status="CHECKING")
                save()
                row.update(restore_evidence(before, after, expected, actual, payload_bytes))
                if args.source_medium == "ssd":
                    row.update(verify_source_ssd(before[0], after[0]))
                    if row["source_ssd_read_bytes"] != payload_bytes:
                        raise AssertionError(
                            "SSD read bytes do not cover the exact requested prefix"
                        )
                row["status"] = "PASS"
                save()
        result["status"] = "VALID_DESCRIPTIVE_PROFILE"
    except BaseException as error:
        result.update(status="INVALID_FAIL_STOP", error=repr(error))
        raise
    finally:
        save()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    for name in ("model", "source-url", "target-url", "source-manager", "target-manager"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prompts", type=external_path, required=True)
    parser.add_argument("--bytes-per-token", type=int, required=True)
    parser.add_argument("--block-tokens", type=int, default=64)
    parser.add_argument("--remote-repeats", type=int, default=10)
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--source-medium", choices=("dram", "ssd"), default="dram")
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if (
        not isinstance(prompts, list)
        or not prompts
        or args.output_tokens <= 0
        or any(
            not isinstance(prompt, list)
            or len(prompt) < args.block_tokens
            or any(type(token) is not int or token < 0 for token in prompt)
            for prompt in prompts
        )
    ):
        parser.error("Expected fresh token-ID arrays with at least one complete block")
    result = profile(args, prompts)
    print(json.dumps({"status": result["status"], "samples": len(result["samples"])}))


if __name__ == "__main__":
    main()
