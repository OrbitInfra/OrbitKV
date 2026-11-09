"""The experiment runner preserves failures, provenance and external-only output."""

import json
from dataclasses import asdict, replace

import pytest

from benches.storage.state_cache.campaign import (
    BASELINES,
    ROOT,
    campaign,
    external_directory,
    matched,
    p3_regions,
    variants,
)
from benches.storage.state_cache.model import LimitError
from benches.storage.state_cache.policy import POLICIES
from benches.storage.state_cache.trace import input_size
from benches.tests.storage.state_cache.fixtures import COSTS, SCHEMA, SIZES


def parameters():
    return {
        "sizes": asdict(SIZES),
        "costs": asdict(replace(COSTS, bytes_per_us=4096)),
        "capacities": {"hbm": 1 << 20, "dram": 1 << 20, "ssd": 1 << 22},
        "workload": {"sessions": 1, "turns": 2, "forks": 1, "pages": 4, "steps": 3},
        "sweep": {"seed": [6, 7]},
    }


def test_campaign_reproduction_provenance_and_failure_artifacts(tmp_path):
    p = parameters()
    p["policies"] = [
        "minimal_lru",
        "p1",
        "p2",
        "p3",
        "oracle",
        {"base": "all", "name": "no_jit", "jit": False},
    ]
    output = tmp_path / "campaign"
    a = campaign(SCHEMA, p, output)
    b = campaign(SCHEMA, p, tmp_path / "repeat")
    assert a == b and all(row["status"] == "ok" for row in a)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_sha256"] and manifest["schema"]["unresolved"]
    assert manifest["simulation_only"] and not manifest["calibrated"]
    assert len((output / "sweep.csv").read_text().splitlines()) == len(a) + 1
    assert (output / "run-0000/events.jsonl").read_text()
    with pytest.raises(FileExistsError):
        campaign(SCHEMA, p, output)
    p["limits"] = {"nodes": 1}
    failed = campaign(SCHEMA, p, tmp_path / "failed")
    assert all(row["status"] == "failed" for row in failed)
    assert (tmp_path / "failed/run-0000/result.json").exists()


def test_external_output_and_nested_reference_caps(tmp_path):
    for path in (ROOT, ROOT / "results/sim"):
        with pytest.raises(ValueError, match="outside"):
            external_directory(path)
    link = tmp_path / "checkout"
    link.symlink_to(ROOT, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        external_directory(link / "sim")
    with pytest.raises(LimitError, match="reference"):
        input_size(tuple(("a",) * 10 for _ in range(10)), 50)
    with pytest.raises(LimitError, match="cap"):
        list(variants(parameters(), 1))
    with pytest.raises(ValueError, match="axes"):
        list(variants({**parameters(), "sweep": {"unknown": [1]}}, 100))


def test_point_gate_needs_all_controls_complete_work_and_measured_latency():
    metrics = {
        "completed": 10,
        "rejected": 0,
        "ttft_p95_us": 100,
        "tpot_p95_us": 10,
        "peak_replicated_bytes": 100,
        "wire_bytes": 100,
    }
    rows = [
        {
            "run": name,
            "status": "ok",
            "settings": {},
            "policy": asdict(POLICIES[name]),
            "metrics": dict(metrics),
        }
        for name in sorted(BASELINES) + ["p1", "oracle"]
    ]
    rows[-2]["metrics"]["peak_replicated_bytes"] = 79
    assert matched(rows)[0]["p1_point_all_controls"]
    assert len(matched(rows)) == 1  # Lookahead control is never an online winner.
    rows[-2]["metrics"]["rejected"] = 1
    assert not matched(rows)[0]["p1_point_all_controls"]
    rows[-2]["metrics"]["rejected"] = 0
    rows[-2]["metrics"]["tpot_p95_us"] = None
    assert not matched(rows)[0]["p1_point_all_controls"]


def test_joint_p3_gate_requires_bytes_capacity_latency_and_oracle_slo():
    rows = []
    for name in [*sorted(BASELINES), "paging", "previous_step", "paging_oracle"]:
        for sessions in (1, 2, 3):
            candidate = name in {"previous_step", "paging_oracle"}
            rows.append(
                {
                    "run": f"{name}-{sessions}",
                    "status": "ok",
                    "policy": asdict(POLICIES[name]),
                    "settings": {"workload": {"layered": True, "sessions": sessions}},
                    "metrics": {
                        "slo_pass": candidate or sessions <= 2,
                        "wire_bytes_per_decode_tokens": 60 if candidate else 100,
                        "ttft_p95_us": 100,
                        "tpot_p95_us": 10,
                        "speculative_bytes": 10 if candidate else 0,
                        "completed": sessions,
                        "decode_tokens": sessions * 10,
                    },
                }
            )
    assert p3_regions(rows)[0]["p3_joint_gate"]
    target = next(r for r in rows if r["run"] == "previous_step-3")
    target["metrics"]["tpot_p95_us"] = 11
    assert not p3_regions(rows)[0]["p3_joint_gate"]
    target["metrics"]["tpot_p95_us"] = 10
    oracle = next(r for r in rows if r["run"] == "paging_oracle-3")
    oracle["metrics"]["slo_pass"] = False
    assert not p3_regions(rows)[0]["p3_joint_gate"]
