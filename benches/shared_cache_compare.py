"""Compare frozen independent serving pairs without treating requests as replicates.

This evaluates a serial SSD read-batch setting. Pressure, cancellation and
broader serving qualification are separate promotion gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from .artifacts import external_path
from .shutdown import validate_component_stops


def summarize(root: Path, contract: dict) -> dict:
    design = contract["design"]
    if design["pairs"] < 3 or design["warmup"] < 1 or design["measured"] < 20:
        raise ValueError("Independent pairs, declared warmup and measured samples are required")
    campaign = json.loads((root / "CAMPAIGN.json").read_text())
    if campaign["status"] != "PASS_COMPONENT_COLLECTION":
        raise ValueError("Incomplete or invalid campaigns cannot establish performance")
    runs = {run["name"]: run for run in campaign["cells"]}
    if (
        len(runs) != len(campaign["cells"])
        or len(runs) != len(contract["cells"])
        or set(runs) != {cell["name"] for cell in contract["cells"]}
    ):
        raise ValueError("Every frozen cell must occur exactly once")
    pairs = {}
    for cell in contract["cells"]:
        name = cell["name"]
        run = runs[name]
        if run["status"] != "PASS_COMPONENT_CELL" or run["spec"] != cell:
            raise ValueError(f"Invalid cell or changed design: {name}")
        validate_component_stops(cell, run.get("stops"))
        path = root / f"{name}-profile.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["profile_sha256"]:
            raise ValueError(f"Changed profile: {name}")
        profile = json.loads(path.read_text())
        if (
            profile["status"] != "VALID_DESCRIPTIVE_PROFILE"
            or profile["engine"] != cell["engine"]
            or profile["source_medium"] != "ssd"
            or profile["warmup_repeats_per_prompt"] != design["warmup"]
            or profile["measured_repeats_per_prompt"] != design["measured"]
            or profile["bytes_per_token"] * profile["block_tokens"] != design["block_bytes"]
            or profile["block_tokens"] != design["block_tokens"]
            or {row["prompt"] for row in profile["samples"]}
            != set(range(len(design["prompt_tokens"])))
        ):
            raise ValueError(f"Profile contract mismatch: {name}")
        for prompt, tokens in enumerate(design["prompt_tokens"]):
            rows = [row for row in profile["samples"] if row["prompt"] == prompt]
            expected = design["warmup"] + design["measured"]
            if len(rows) != expected or [row["repeat"] for row in rows] != list(range(expected)):
                raise ValueError(f"Missing, duplicate or reordered samples: {name}/{prompt}")
            payload = tokens // design["block_tokens"] * design["block_bytes"]
            batch_blocks = max(1, cell["batch_mib"] * 1024**2 // design["block_bytes"])
            expected_batches = math.ceil(payload / design["block_bytes"] / batch_blocks)
            for row in rows:
                if (
                    row["status"] != "PASS"
                    or row["sample_kind"]
                    != ("warmup" if row["repeat"] < design["warmup"] else "measured")
                    or not row["output_match"]
                    or row["remote_bytes"] != payload
                    or row["h2d_bytes"] != payload
                    or row["source_ssd_read_bytes"] != payload
                    or row["source_response"]["usage"]["prompt_tokens"] != tokens
                    or row["remote_stages"]["authorization"]["calls"] != expected_batches
                    or row["remote_stages"]["read"]["calls"] != expected_batches
                ):
                    raise ValueError(
                        f"Incorrect payload or unconsumed batch setting: {name}/{prompt}"
                    )
            measured = rows[design["warmup"] :]
            key = (cell["engine"], prompt, cell["pair"])
            pair = pairs.setdefault(key, {})
            if cell["variant"] in pair:
                raise ValueError(f"Duplicate variant: {name}")
            pair[cell["variant"]] = {
                "order": cell["order_index"],
                "output": measured[0]["source_response"],
                "model": profile["model"],
                "prompt_token_ids_sha256": profile["prompt_token_ids_sha256"],
                "ttft_ms": statistics.median(row["ttft_ms"] for row in measured),
                "e2e_ms": statistics.median(row["e2e_ms"] for row in measured),
                "authorization_ms": statistics.median(
                    row["remote_stages"]["authorization"]["total_ms"] for row in measured
                ),
                "read_ms": statistics.median(
                    row["remote_stages"]["read"]["total_ms"] for row in measured
                ),
                "authorization_calls": expected_batches,
                "samples": design["measured"],
            }
    summaries = []
    for engine in design["engines"]:
        for prompt, tokens in enumerate(design["prompt_tokens"]):
            group = []
            for pair_index in range(design["pairs"]):
                pair = pairs[engine, prompt, pair_index]
                baseline, candidate = pair["baseline"], pair["candidate"]
                expected_order = (
                    ["baseline", "candidate"] if pair_index % 2 == 0 else ["candidate", "baseline"]
                )
                if sorted(pair, key=lambda variant: pair[variant]["order"]) != expected_order:
                    raise ValueError("Pair order must alternate")
                if any(
                    baseline[field] != candidate[field]
                    for field in ("model", "prompt_token_ids_sha256")
                ):
                    raise ValueError("Matched pair used different model or prompt token IDs")
                if baseline["output"]["text"] != candidate["output"]["text"]:
                    raise ValueError("Matched pair has different native output")
                for field in ("prompt_tokens", "completion_tokens"):
                    if baseline["output"]["usage"][field] != candidate["output"]["usage"][field]:
                        raise ValueError("Matched pair has different native token counts")
                group.append(
                    {
                        "pair": pair_index,
                        "baseline": baseline,
                        "candidate": candidate,
                        "ratios": {
                            metric: candidate[metric] / baseline[metric]
                            for metric in ("ttft_ms", "e2e_ms")
                        },
                    }
                )
            metrics = {}
            for metric in ("ttft_ms", "e2e_ms"):
                ratios = [pair["ratios"][metric] for pair in group]
                rng = random.Random(design["bootstrap_seed"])
                estimates = sorted(
                    math.exp(statistics.fmean(math.log(rng.choice(ratios)) for _ in ratios))
                    for _ in range(design["bootstrap_draws"])
                )
                metrics[metric] = {
                    "geometric_mean_ratio": math.exp(statistics.fmean(map(math.log, ratios))),
                    "bootstrap_95_ci": [
                        estimates[round((len(estimates) - 1) * fraction)]
                        for fraction in (0.025, 0.975)
                    ],
                    "maximum_pair_ratio": max(ratios),
                }
            guards = contract["guards"]
            passed = (
                metrics["ttft_ms"]["geometric_mean_ratio"] <= guards["ttft_ratio"]
                and metrics["ttft_ms"]["bootstrap_95_ci"][1] < guards["ttft_ci_upper"]
                and all(
                    value["maximum_pair_ratio"] <= guards["maximum_pair_ratio"]
                    for value in metrics.values()
                )
            )
            summaries.append(
                {
                    "engine": engine,
                    "prompt_tokens": tokens,
                    "pairs": group,
                    "metrics": metrics,
                    "component_passed": passed,
                }
            )
    return {
        "state": "VALID_COMPONENT_PASS"
        if all(row["component_passed"] for row in summaries)
        else "VALID_COMPONENT_FAIL",
        "statistical_unit": "independent matched process pair; requests are not resampled",
        "summary": summaries,
        "production_qualified": False,
        "remaining_promotion_gates": ["matched pressure", "cancellation and drain"],
        "p99_qualified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=external_path, required=True)
    parser.add_argument("--contract", type=external_path, required=True)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve the existing analysis")
    result = summarize(args.root, json.loads(args.contract.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state": result["state"], "production_qualified": False}))


if __name__ == "__main__":
    main()
