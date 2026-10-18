"""Summarize independent quiet/metadata-pressure S2.10 run pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from benches.cluster.live_store_measurements import MEASUREMENT_CONTRACT, _context_switch_delta
from benches.common.artifacts import external_path

PRIMARY_METRICS = ("manager_save_submit_ms", "local_query_ms")


def _geometric_mean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _paired_bootstrap(values, samples=10_000):
    rng = random.Random(20261002)
    estimates = sorted(
        _geometric_mean([rng.choice(values) for _ in values]) for _ in range(samples)
    )
    return {
        "statistic": "paired-run geometric mean ratio",
        "samples": samples,
        "lower": estimates[round((samples - 1) * 0.025)],
        "upper": estimates[round((samples - 1) * 0.975)],
    }


def summarize(root: Path, media, seeds, qualification):
    assert len(seeds) == len(set(seeds)) and seeds
    if qualification:
        assert len(seeds) >= 5
    threshold = 1.05
    result = {
        "measurement_contract": MEASUREMENT_CONTRACT,
        "statistical_unit": "independent matched run pair",
        "request_samples_are_not_bootstrap_units": True,
        "threshold_ratio": threshold,
        "qualification": qualification,
        "media": {},
    }
    passed = True
    for medium in media:
        pairs = []
        ratios = {metric: [] for metric in PRIMARY_METRICS}
        for pair_index, seed in enumerate(seeds):
            runs = {}
            for condition in ("quiet", "pressure"):
                path = root / f"{medium}-{seed}-{condition}" / "isolation-result.json"
                run = json.loads(path.read_text())
                assert run["measurement_contract"] == MEASUREMENT_CONTRACT
                assert run["status"] == "passed"
                assert run["condition"] == condition
                assert run["medium"] == medium
                assert run["seed"] == seed
                assert run["samples"] == run["latencies"]["local_query_ms"]["samples"]
                assert 0.95 <= run["achieved_samples_per_second"] * run["cadence_ms"] / 1000 <= 1.05
                if qualification:
                    assert run["samples"] >= 1000
                    assert run["warmup_rounds"] >= 50
                runs[condition] = run
            quiet, pressure = runs["quiet"], runs["pressure"]
            assert quiet["pressure_result"]["mode"] == "quiet"
            assert quiet["pressure_result"]["changes"] == 0
            assert pressure["pressure_result"]["mode"] == "pressure"
            assert pressure["pressure_result"]["changes"] == (
                pressure["pressure_rounds"] * pressure["pressure_window_shift"] * 2
            )
            for field in (
                "payload_schema",
                "samples",
                "warmup_rounds",
                "cadence_ms",
                "local_namespace",
                "pressure_namespace",
                "pressure_keys",
                "pressure_active_keys",
                "pressure_window_shift",
                "pressure_profile",
                "pressure_rounds",
                "pressure_cadence_ms",
                "observer_sample_ms",
                "artifacts",
                "budgets",
                "key_oracle_sha256",
                "payload_oracle_sha256",
            ):
                assert quiet[field] == pressure[field], (field, quiet[field], pressure[field])
            if pair_index % 2 == 0:
                assert quiet["order_index"] < pressure["order_index"]
            else:
                assert pressure["order_index"] < quiet["order_index"]
            pair = {
                "seed": seed,
                "order": [
                    condition
                    for condition, _ in sorted(
                        (("quiet", quiet["order_index"]), ("pressure", pressure["order_index"])),
                        key=lambda item: item[1],
                    )
                ],
                "samples": quiet["samples"],
                "quiet": {metric: quiet["latencies"][metric] for metric in PRIMARY_METRICS},
                "pressure": {metric: pressure["latencies"][metric] for metric in PRIMARY_METRICS},
                "ratios": {},
                "absolute_p99_delta_ms": {},
                "resource_delta": {
                    condition: {
                        "cpu_ticks": run["process_after"]["cpu_ticks"]
                        - run["process_before"]["cpu_ticks"],
                        "context_switches": _context_switch_delta(
                            run["process_before"], run["process_after"]
                        ),
                        "rss_before_kib": run["process_before"].get("rss_kib"),
                        "rss_after_kib": run["process_after"].get("rss_kib"),
                    }
                    for condition, run in runs.items()
                },
            }

            for metric in PRIMARY_METRICS:
                quiet_p99 = quiet["latencies"][metric]["p99"]
                pressure_p99 = pressure["latencies"][metric]["p99"]
                ratio = pressure_p99 / quiet_p99
                ratios[metric].append(ratio)
                pair["ratios"][metric] = ratio
                pair["absolute_p99_delta_ms"][metric] = pressure_p99 - quiet_p99
            pair["every_primary_ratio_at_most_1_05"] = all(
                ratio <= threshold for ratio in pair["ratios"].values()
            )
            pairs.append(pair)
        metric_results = {}
        for metric, values in ratios.items():
            interval = _paired_bootstrap(values)
            metric_results[metric] = {
                "per_pair_ratios": values,
                "geometric_mean_ratio": _geometric_mean(values),
                "paired_bootstrap_95_percent_ci": interval,
                "every_pair_at_most_1_05": max(values) <= threshold,
                "ci_upper_at_most_1_05": interval["upper"] <= threshold,
            }
        medium_passed = all(
            row["every_pair_at_most_1_05"] and row["ci_upper_at_most_1_05"]
            for row in metric_results.values()
        )
        passed &= medium_passed
        result["media"][medium] = {
            "pairs": pairs,
            "metrics": metric_results,
            "isolation_qualified": qualification and medium_passed,
        }
    result["status"] = ("passed" if passed else "failed") if qualification else "preexperiment"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=external_path, required=True)
    parser.add_argument("--media", nargs="+", choices=("dram", "ssd"), required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    if args.qualification and len(args.seeds) < 5:
        parser.error("qualification requires at least five independent seeds")
    result = summarize(args.root, args.media, args.seeds, args.qualification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
