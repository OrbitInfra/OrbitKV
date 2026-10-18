import json

import pytest

from benches.cluster.live_store_measurements import _installation_sample, _process_sample
from benches.cluster.summarize_live_store_isolation import summarize


def test_installation_sample_uses_owner_commit_timestamps_not_http_completion():
    targets = {
        "a": {
            "node": "a",
            "incarnation": "owner-a",
            "view_id": "view-a",
            "sequence": 9,
            "published_mono_ns": 1_000_000,
            "save_start_mono_ns": 500_000,
        },
        "b": {
            "node": "b",
            "incarnation": "owner-b",
            "view_id": "view-b",
            "sequence": 11,
            "published_mono_ns": 2_000_000,
            "save_start_mono_ns": 1_250_000,
        },
    }
    installed = {
        "a": {
            "applied_sequence": 9,
            "view_id": "view-a",
            "installed_mono_ns": 4_000_000,
        },
        "b": {
            "applied_sequence": 11,
            "view_id": "view-b",
            "installed_mono_ns": 5_500_000,
        },
    }
    sample = _installation_sample(targets, installed, 8_000_000)
    assert sample["publication_to_install_ms"] == 3.5
    assert sample["save_start_to_install_ms"] == 4.25
    assert sample["install_to_harness_ms"] == 4.0
    assert sample["owners"][0]["installed_sequence"] == 9
    assert sample["owners"][1]["installed_sequence"] == 11
    installed["b"]["applied_sequence"] = 12
    with pytest.raises(AssertionError):
        _installation_sample(targets, installed, 8_000_000)


def _run(condition, medium, seed, order, p99):
    summary = {"samples": 20, "p50": p99 / 2, "p95": p99 * 0.9, "p99": p99, "max": p99}
    return {
        "measurement_contract": "s2.10-performance-v2",
        "payload_schema": "u64-generation-u64-block-le-v1",
        "status": "passed",
        "condition": condition,
        "medium": medium,
        "seed": seed,
        "samples": 20,
        "warmup_rounds": 2,
        "cadence_ms": 1000,
        "achieved_samples_per_second": 1.0,
        "local_namespace": "local",
        "pressure_namespace": "pressure",
        "pressure_keys": 32,
        "pressure_active_keys": 24,
        "pressure_window_shift": 4,
        "pressure_profile": "sustained-17ms-v1",
        "pressure_rounds": 1320,
        "pressure_cadence_ms": 17,
        "observer_sample_ms": 25,
        "artifacts": {"manager": {"sha256": "same"}},
        "order_index": order,
        "budgets": {},
        "key_oracle_sha256": "keys",
        "payload_oracle_sha256": "payloads",
        "latencies": {
            "manager_save_submit_ms": summary,
            "local_query_ms": summary,
        },
        "process_before": {
            "thread_context_switches": {},
            "cpu_ticks": 1,
            "voluntary_context_switches": 1,
            "nonvoluntary_context_switches": 1,
        },
        "process_after": {
            "thread_context_switches": {},
            "cpu_ticks": 2,
            "voluntary_context_switches": 2,
            "nonvoluntary_context_switches": 2,
        },
        "pressure_result": {
            "mode": condition,
            "changes": 1320 * 4 * 2 if condition == "pressure" else 0,
        },
    }


def test_isolation_summary_pairs_runs_and_never_uses_requests_as_bootstrap_units(tmp_path):
    seeds = ["one", "two"]
    for pair, seed in enumerate(seeds):
        order = (pair * 2, pair * 2 + 1)
        if pair % 2:
            order = order[::-1]
        for condition, index in zip(("quiet", "pressure"), order, strict=True):
            directory = tmp_path / f"dram-{seed}-{condition}"
            directory.mkdir()
            p99 = 1.0 if condition == "quiet" else 1.04
            (directory / "isolation-result.json").write_text(
                json.dumps(_run(condition, "dram", seed, index, p99))
            )
    result = summarize(tmp_path, ["dram"], seeds, False)
    assert result["statistical_unit"] == "independent matched run pair"
    assert result["request_samples_are_not_bootstrap_units"]
    assert result["media"]["dram"]["metrics"]["local_query_ms"]["every_pair_at_most_1_05"]
    assert result["status"] == "preexperiment"
    assert not result["media"]["dram"]["isolation_qualified"]


def test_process_sample_reports_context_switch_and_affinity_fields():
    sample = _process_sample([__import__("os").getpid()])
    assert sample["processes"] == 1
    assert sample["cpu_affinity"]
    assert sample["memory_affinity"]
    assert sample["voluntary_context_switches"] >= 0
    assert sample["nonvoluntary_context_switches"] >= 0
    assert sample["thread_cpu_ticks"]
    assert sample["sample_cpu_ns"] > 0
    assert sample["sample_elapsed_ns"] >= sample["sample_cpu_ns"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--samples", "20", "--warmup-cycles", "50"],
        ["--samples", "1000", "--warmup-cycles", "0"],
        ["--samples", "1000", "--warmup-cycles", "50", "--skip-restore"],
    ],
)
def test_capacity_cli_rejects_unqualified_samples_before_runtime(tmp_path, arguments):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "python") + os.pathsep + str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benches.cluster.live_store_capacity",
            "--owners",
            "16",
            "--seed",
            "guard",
            "--enforce-thresholds",
            "--output",
            str(tmp_path / "run"),
            *arguments,
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "qualification requires" in result.stderr
    assert not (tmp_path / "run").exists()


def test_pressure_exposure_rejects_phase_lock_and_missing_observer_progress(tmp_path):
    from benches.cluster.live_store_measurements import _pressure_exposure

    base = 1_000_000_000
    source = []
    observer = []
    for step in range(30):
        first = base + step * 20_000_000
        source.append(
            {
                "actual_seconds": step * 0.020,
                "scheduled_seconds": step * 0.020,
                "inventory": {"sequence": (step + 1) * 2},
                "sequence_before": step * 2,
                "scheduled_mono_ns_lower": first,
                "scheduled_mono_ns_upper": first,
                "first_publication_mono_ns": first,
                "last_publication_mono_ns": first + 100_000,
            }
        )
        observer.append(
            {
                "poll_start_mono_ns": first + 2_000_000,
                "poll_end_mono_ns": first + 3_000_000,
                "sampler_cpu_ns": 100,
                "owner": {
                    "owner": "source",
                    "view_id": "view",
                    "applied_sequence": step + 1,
                    "installed_mono_ns": first + 1_000_000,
                },
            }
        )
    foreground = []
    for step in range(4):
        start = base + 2_000_000 + step * 125_000_000
        foreground.append(
            {
                "round": step,
                "measured": True,
                "round_start_mono_ns": start,
                "save_start_mono_ns": start,
                "save_end_mono_ns": start + 100_000,
                "query_start_mono_ns": start + 200_000,
                "query_end_mono_ns": start + 500_000,
                "gpu_completed_mono_ns": start + 600_000,
            }
        )

    def write(name, values):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in values))

    write("samples.jsonl", foreground)
    write("pressure-samples.jsonl", source)
    write("observer-samples.jsonl", observer)
    (tmp_path / "pressure-source-result.json").write_text(json.dumps({"elapsed_seconds": 0.6}))
    report = _pressure_exposure(tmp_path, "pressure", 30, 20, 100, 1)
    assert report["offered_phase_quarters"] == [0, 1, 2, 3]
    assert not report["windows_without_observed_install"]
    source[-1]["first_publication_mono_ns"] += 120_000_000
    source[-1]["last_publication_mono_ns"] += 120_000_000
    write("pressure-samples.jsonl", source)
    with pytest.raises(AssertionError):
        _pressure_exposure(tmp_path, "pressure", 30, 20, 100, 1)
    source[-1]["first_publication_mono_ns"] -= 120_000_000
    source[-1]["last_publication_mono_ns"] -= 120_000_000
    write("pressure-samples.jsonl", source)
    for row in observer:
        row["owner"]["installed_mono_ns"] = base - 1
    write("observer-samples.jsonl", observer)
    with pytest.raises(AssertionError):
        _pressure_exposure(tmp_path, "pressure", 30, 20, 100, 1)


def test_payload_header_disambiguates_periodic_body_and_bounds_identity():
    from benches.cluster.live_store_measurements import _payload, _payload_header

    first = _payload_header(1, 7)
    aliased_body = _payload_header(252, 7)
    assert first != aliased_body
    assert int.from_bytes(first[:8], "little") == 1
    assert int.from_bytes(first[8:], "little") == 7
    assert first != _payload_header(1, 8)
    assert len(_payload_header(2**64 - 1, 2**64 - 1)) == 16
    with pytest.raises(ValueError):
        _payload(None, 1, 15, 1)
    for generation, block in [(-1, 0), (2**64, 0), (0, -1), (0, 2**64)]:
        with pytest.raises(ValueError):
            _payload_header(generation, block)
