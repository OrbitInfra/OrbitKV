"""External-only artifacts and matched, explicitly simulated parameter sweeps."""

import csv
import hashlib
import itertools
import json
import platform
import subprocess
from dataclasses import asdict, fields, replace
from pathlib import Path

from .engine import Costs, Simulation
from .model import LimitError, Limits
from .policy import POLICIES
from .trace import Sizes, Workload, generate

ROOT = Path(__file__).resolve().parents[3]
BASELINES = {"full_lru", "minimal_lru", "shared_lru", "static"}


def external_directory(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("simulation artifacts must be outside the source checkout")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def variants(parameters, maximum):
    if not 1 <= maximum <= 4096:
        raise LimitError("campaign run cap must be in [1, 4096]")
    sweep = parameters.get("sweep", {})
    allowed = {f.name for cls in (Workload, Sizes, Costs) for f in fields(cls)} | {
        "hbm_bytes",
        "dram_bytes",
        "ssd_bytes",
    }
    if set(sweep) - allowed or any(
        not isinstance(v, list) or not 1 <= len(v) <= 256 for v in sweep.values()
    ):
        raise ValueError("invalid sweep axes or axis length")
    policies = parameters.get("policies", list(POLICIES))
    if not isinstance(policies, list) or not policies or len(policies) > 32:
        raise LimitError("policy count budget")
    count = len(policies)
    for choices in sweep.values():
        count *= len(choices)
    if count > maximum:
        raise LimitError(f"{count} runs exceed campaign cap {maximum}")
    for values in itertools.product(*sweep.values()):
        axes = dict(zip(sweep, values, strict=True))
        settings = {}
        for name, cls in (("sizes", Sizes), ("costs", Costs), ("workload", Workload)):
            settings[name] = cls(
                **{
                    **parameters.get(name, {}),
                    **{k: v for k, v in axes.items() if k in {f.name for f in fields(cls)}},
                }
            )
        settings["capacities"] = {
            tier: axes.get(f"{tier}_bytes", parameters["capacities"][tier])
            for tier in ("hbm", "dram", "ssd")
        }
        for spec in policies:
            if isinstance(spec, str):
                policy = POLICIES[spec]
            else:
                spec = dict(spec)
                policy = replace(POLICIES[spec.pop("base")], **spec)
            if len(policy.name) > 64 or not policy.name.replace("_", "").isalnum():
                raise ValueError("invalid policy label")
            yield axes, policy, settings


def matched(rows):
    """Strongest compatible *approximation*, never product benchmark evidence.

    Return pointwise gates; promotion additionally requires repeated seeds, two
    holdout families and adjacent budgets. Session capacity needs a concurrency
    sweep; a single run cannot assert a 20% capacity improvement.
    """
    groups = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        signature = json.dumps(row["settings"], sort_keys=True)
        groups.setdefault(signature, []).append(row)
    results = []
    for group in groups.values():
        baseline = [r for r in group if r["policy"]["name"] in BASELINES]
        if len(baseline) != len(BASELINES):
            continue
        for row in group:
            if row["policy"]["name"] in BASELINES or row["policy"]["oracle"]:
                continue
            m = row["metrics"]
            checks = []
            for b in baseline:
                bm = b["metrics"]
                complete = (
                    m["completed"] > 0
                    and all(
                        x[metric] is not None
                        for x in (m, bm)
                        for metric in ("ttft_p95_us", "tpot_p95_us")
                    )
                    and m["completed"] == bm["completed"]
                    and not m["rejected"]
                    and not bm["rejected"]
                )
                ttft = (m["ttft_p95_us"] or 0) / max(1, bm["ttft_p95_us"] or 0)
                tpot = (m["tpot_p95_us"] or 0) / max(1, bm["tpot_p95_us"] or 0)
                storage = m["peak_replicated_bytes"] / max(1, bm["peak_replicated_bytes"])
                wire = m["wire_bytes"] / max(1, bm["wire_bytes"])
                checks.append(
                    {
                        "baseline": b["policy"]["name"],
                        "ttft_ratio": ttft,
                        "tpot_ratio": tpot,
                        "stored_ratio": storage,
                        "wire_ratio": wire,
                        "p1_point": complete
                        and tpot <= 1.05
                        and ((storage <= 0.8 and ttft <= 1) or ttft <= 0.9),
                        "p3_bytes_point": complete
                        and wire <= 0.7
                        and ttft <= 1.05
                        and tpot <= 1.05,
                    }
                )
            results.append(
                {
                    "run": row["run"],
                    "policy": row["policy"]["name"],
                    "comparisons": checks,
                    "p1_point_all_controls": all(c["p1_point"] for c in checks),
                    "p3_bytes_point_all_controls": all(c["p3_bytes_point"] for c in checks),
                }
            )
    return results


def capacity_table(rows):
    groups = {}
    for row in rows:
        settings = {**row["settings"], "workload": dict(row["settings"]["workload"])}
        sessions = settings["workload"].pop("sessions")
        key = json.dumps({"settings": settings, "policy": row["policy"]}, sort_keys=True)
        entry = groups.setdefault(
            key,
            {
                "settings": settings,
                "policy": row["policy"],
                "tested_sessions": [],
                "slo_admitted_sessions_per_modeled_gpu": 0,
            },
        )
        entry["tested_sessions"].append(sessions)
        if row["status"] == "ok" and row["metrics"]["slo_pass"]:
            entry["slo_admitted_sessions_per_modeled_gpu"] = max(
                entry["slo_admitted_sessions_per_modeled_gpu"], sessions
            )
    for entry in groups.values():
        entry["tested_sessions"] = sorted(set(entry["tested_sessions"]))
        entry["capacity_is_lower_bound"] = entry["slo_admitted_sessions_per_modeled_gpu"] == max(
            entry["tested_sessions"]
        )
        entry["capacity_gain_established"] = (
            False  # Matching latency + replicated holdout gates are still required.
        )
    return list(groups.values())


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for name in sorted({r["policy"]["name"] for r in rows}):
        points = [r for r in rows if r["status"] == "ok" and r["policy"]["name"] == name]
        ax.scatter(
            [r["settings"]["capacities"]["hbm"] / 2**20 for r in points],
            [r["metrics"]["wire_bytes_per_decode_tokens"] for r in points],
            label=name,
            s=18,
            alpha=0.65,
        )
    ax.set(
        xlabel="Modeled HBM budget (MiB)",
        ylabel="Wire bytes / completed decode token",
        title="Simulation only; uncalibrated costs.\nAll trials, including rejected requests and SLO misses.",
    )
    ax.legend(fontsize=7, ncols=3)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def campaign(schema, parameters, output, *, maximum=512, plots=False):
    runs = list(variants(parameters, maximum))
    limits = Limits(**parameters.get("limits", {}))
    output = external_directory(output)
    git = [
        subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        for args in (("rev-parse", "HEAD"), ("status", "--porcelain"))
    ]
    revision, dirty = (r.stdout.strip() if r.returncode == 0 else "unavailable" for r in git)
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    write_json(
        output / "manifest.json",
        {
            "revision": revision,
            "dirty": dirty,
            "git_metadata_available": all(r.returncode == 0 for r in git),
            "source_sha256": source_hashes,
            "python": platform.python_version(),
            "schema": schema.record(),
            "parameters": parameters,
            "limits": asdict(limits),
            "simulation_only": True,
            "calibrated": False,
            "baseline_scope": "approximate controls, not vLLM/LMCache/HiCache executions",
            "clock": "integer microseconds; conservative shared link and compute lane",
        },
    )
    rows = []
    for index, (axes, policy, settings) in enumerate(runs):
        run = f"run-{index:04d}"
        directory = output / run
        directory.mkdir()
        row = {
            "run": run,
            "axes": axes,
            "policy": asdict(policy),
            "settings": {k: asdict(v) if k != "capacities" else v for k, v in settings.items()},
        }
        try:
            trace = generate(schema, settings["sizes"], settings["workload"], limits)
            write_json(directory / "trace.json", asdict(trace))
            with (directory / "events.jsonl").open("w") as log:

                def emit(event):
                    log.write(json.dumps(event, sort_keys=True) + "\n")

                result = Simulation(
                    trace, policy, settings["capacities"], settings["costs"], limits, emit
                ).run()
            row.update(status="ok", metrics=result)
        except (ValueError, KeyError) as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(directory / "result.json", row)
        rows.append(row)
    write_json(output / "runs.json", rows)
    write_json(output / "point-gates.json", matched(rows))
    write_json(output / "capacity.json", capacity_table(rows))
    flat = [
        {
            "run": r["run"],
            "status": r["status"],
            "policy": r["policy"]["name"],
            **r["axes"],
            **{k: v for k, v in r.get("metrics", {}).items() if not isinstance(v, (dict, list))},
        }
        for r in rows
    ]
    with (output / "sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in flat))))
        writer.writeheader()
        writer.writerows(flat)
    if plots:
        plot(rows, output / "wire-budget.png")
    return rows
