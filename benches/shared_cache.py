"""Qualify independent matching replicas through their existing serving APIs.

The Managers own discovery, authorization, transfer and source reservations in
Rust. This driver submits token-exact requests and checks their exported evidence.
Run only against idle, dedicated qualification deployments with fresh prefixes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from .artifacts import external_path
from .metrics import DRAIN_GAUGES, REMOTE_STAGES, delta, metrics, validate_ownership
from .workload import evict_host_cache, generate

IDLE_METRICS = (
    "orbitkv_query_reserved_bytes",
    "orbitkv_inflight_bytes",
    "orbitkv_transfer_lock_active",
    "orbitkv_transfer_reserved_bytes",
    "orbitkv_transfer_completion_outstanding",
    "orbitkv_ssd_prefetch_inflight",
    "orbitkv_ssd_read_pinned_bytes",
    "orbitkv_ssd_write_queue_pending",
    "orbitkv_ssd_write_inflight",
)


def drain(*manager_urls: str, timeout: float = 30) -> list[dict]:
    deadline = time.monotonic() + timeout
    quiet = 0
    while True:
        snapshots = [metrics(url) for url in manager_urls]
        for snapshot in snapshots:
            validate_ownership(snapshot, drained=False)
            if "orbitkv_transfer_completion_outstanding" not in snapshot:
                raise ValueError("Missing OrbitKV transfer completion ownership snapshot")
        busy = [
            {name: snapshot[name] for name in DRAIN_GAUGES if snapshot.get(name, 0)}
            for snapshot in snapshots
        ]
        quiet = 0 if any(busy) else quiet + 1
        if quiet == 3:
            return snapshots
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Shared-cache resources did not drain: {busy}")
        time.sleep(0.1)


def synchronize(manager_url: str, *readers: str, timeout: float = 30) -> dict:
    response = requests.post(f"{manager_url}/cache/sync", timeout=35)
    response.raise_for_status()
    fence = response.json()["inventory_fence"]
    for reader in readers:
        response = requests.get(f"{reader}/cache/metadata", timeout=5)
        response.raise_for_status()
        status = response.json()
        response = requests.post(
            f"{reader}/cache/metadata/await",
            json={
                "inventory_fence": fence,
                "scope_digest": status["stream"]["scope_digest"],
                "timeout_ms": int(timeout * 1000),
            },
            timeout=timeout + 5,
        )
        if response.status_code != 200:
            raise TimeoutError(f"{reader} did not install inventory fence {fence}: {response.text}")
    return fence


def verify_restore(before: dict, after: dict, expected: str, actual: dict) -> dict:
    changes = delta(before, after)
    remote = changes.get("orbitkv_remote_fetch_bytes_total", 0)
    restored = changes.get("orbitkv_load_bytes_total", 0)
    if remote <= 0 or restored <= 0:
        raise AssertionError(f"Expected both Mooncake READ and GPU restore bytes: {changes}")
    if changes.get("orbitkv_remote_stage_duration_seconds_count_release", 0) <= 0:
        raise AssertionError("Expected acknowledged transfer completion evidence")
    if actual["text"] != expected:
        raise AssertionError("Shared-cache output differs from the cold source control")
    return {
        "ttft_ms": actual["ttft_ms"],
        "e2e_ms": actual["e2e_ms"],
        "remote_bytes": int(remote),
        "h2d_bytes": int(restored),
        "remote_stages": {
            stage: {
                "calls": int(
                    changes.get(f"orbitkv_remote_stage_duration_seconds_count_{stage}", 0)
                ),
                "total_ms": changes.get(f"orbitkv_remote_stage_duration_seconds_sum_{stage}", 0)
                * 1000,
            }
            for stage in REMOTE_STAGES
        },
        "output_match": True,
    }


def verify_source_ssd(before: dict, after: dict) -> dict:
    changes = delta(before, after)
    read_bytes = changes.get("orbitkv_ssd_prefetch_bytes_total", 0)
    successes = changes.get("orbitkv_ssd_prefetch_success_total", 0)
    if read_bytes <= 0 or successes <= 0:
        raise AssertionError(f"Expected source SSD materialization before Mooncake READ: {changes}")
    return {
        "source_ssd_read_bytes": int(read_bytes),
        "source_ssd_reads": int(successes),
    }


def qualify(
    *,
    engine: str,
    model: str,
    source_url: str,
    target_url: str,
    source_manager: str,
    target_manager: str,
    prompts: list[list[int]],
    output_tokens: int = 8,
    source_medium: str = "dram",
) -> list[dict]:
    if source_url.rstrip("/") == target_url.rstrip("/") or source_manager.rstrip(
        "/"
    ) == target_manager.rstrip("/"):
        raise ValueError("Shared-cache qualification requires two engines and two Managers")
    if source_medium not in {"dram", "ssd"}:
        raise ValueError("source_medium must be dram or ssd")
    results = []
    for index, prompt in enumerate(prompts):
        source_before, _ = drain(source_manager, target_manager)
        cold = generate(source_url, engine, model, prompt, output_tokens)
        synchronize(source_manager, target_manager)
        source_after, target_before = drain(source_manager, target_manager)
        saved = source_after.get("orbitkv_save_bytes_total", 0) - source_before.get(
            "orbitkv_save_bytes_total", 0
        )
        if saved <= 0:
            raise AssertionError("Source did not publish new pages; use fresh prefixes")
        if source_after.get("orbitkv_load_bytes_total", 0) != source_before.get(
            "orbitkv_load_bytes_total", 0
        ):
            raise AssertionError("Source control unexpectedly restored cached pages")
        preparation = None
        source_transfer_before = source_after
        if source_medium == "ssd":
            ssd_written = source_after.get("orbitkv_ssd_write_bytes_total", 0) - source_before.get(
                "orbitkv_ssd_write_bytes_total", 0
            )
            if ssd_written <= 0:
                raise AssertionError(
                    "Source did not commit SSD bytes; use write policy all and fresh prefixes"
                )
            preparation = evict_host_cache(source_manager)
            synchronize(source_manager, target_manager)
            source_transfer_before, target_before = drain(source_manager, target_manager)
        restored = generate(target_url, engine, model, prompt, output_tokens)
        source_transfer_after, target_after = drain(source_manager, target_manager)
        row = verify_restore(target_before, target_after, cold["text"], restored)
        if source_medium == "ssd":
            row.update(verify_source_ssd(source_transfer_before, source_transfer_after))
        row.update(
            prompt=index,
            input_tokens=len(prompt),
            source_ttft_ms=cold["ttft_ms"],
            save_bytes=int(saved),
            source_medium=source_medium,
            source_preparation=preparation,
            resources_drained=True,
        )
        results.append(row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument(
        "--model", required=True, help="Identical served model name on both replicas"
    )
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--source-manager", required=True)
    parser.add_argument("--target-manager", required=True)
    parser.add_argument(
        "--prompts", type=Path, required=True, help="JSON array of fresh token-ID arrays"
    )
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument(
        "--source-medium",
        choices=("dram", "ssd"),
        default="dram",
        help="Require ordinary DRAM export or force source DRAM eviction and prove SSD staging",
    )
    parser.add_argument(
        "--deployment",
        choices=("same-host-tcp", "two-host-tcp", "two-host-rdma"),
        required=True,
        help="Operator-declared environment; this driver does not detect physical hosts or RDMA",
    )
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(
            not isinstance(prompt, list)
            or len(prompt) < 128
            or any(type(token) is not int or token < 0 for token in prompt)
            for prompt in prompts
        )
        or args.output_tokens < 1
    ):
        parser.error(
            "provide nonempty prompts of at least 128 token IDs and positive output tokens"
        )
    results = qualify(
        engine=args.engine,
        model=args.model,
        source_url=args.source_url,
        target_url=args.target_url,
        source_manager=args.source_manager,
        target_manager=args.target_manager,
        prompts=prompts,
        output_tokens=args.output_tokens,
        source_medium=args.source_medium,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "engine": args.engine,
                "model": args.model,
                "deployment_declared": args.deployment,
                "source_medium": args.source_medium,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified {len(results)} remote restores with drained resources: {args.output}")


if __name__ == "__main__":
    main()
