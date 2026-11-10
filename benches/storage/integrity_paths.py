"""Pair actual native cache paths from two revisions; all output stays external."""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

NODE = "peer::read::tests::path_bench::cpu_paths"
PATHS = {
    "dram_hit",
    "ssd_read",
    "ssd_write_ephemeral",
    "ssd_write_persistent",
    "peer_tcp_read",
}
SIZES = {4096, 1 << 20, 16 << 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ("baseline", "candidate"):
        parser.add_argument(f"--{arm}", required=True, type=Path)
        parser.add_argument(f"--{arm}-revision", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=5, choices=range(1, 22))
    args = parser.parse_args()
    if os.environ.get("ORBITE_REQUIRE_IO_URING") != "1":
        parser.error("ORBITE_REQUIRE_IO_URING=1 is required")
    fs = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", str(args.root)], text=True
    ).strip()
    if fs != "ext4":
        parser.error(f"ext4 required, found {fs}")
    checkout = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(checkout):
        parser.error("output must be outside the source checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evidence": "CPU-only paths, not hardware serving qualification",
        "platform": platform.platform(),
        "filesystem": fs,
        "root": str(args.root),
        "pairs": args.pairs,
        "command": ["--exact", NODE, "--ignored", "--nocapture"],
        "arms": {},
    }
    for arm in ("baseline", "candidate"):
        binary = getattr(args, arm).resolve(strict=True)
        digest = hashlib.sha256()
        with binary.open("rb") as source:
            while chunk := source.read(1 << 20):
                digest.update(chunk)
        manifest["arms"][arm] = {
            "binary": str(binary),
            "sha256": digest.hexdigest(),
            "revision": getattr(args, f"{arm}_revision"),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pairs = []
    for pair in range(args.pairs):
        observed = {}
        for arm in ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline"):
            output = args.output / f"{pair}-{arm}.csv"
            env = dict(
                os.environ,
                ORBITKV_SSD_TEST_ROOT=str(args.root),
                ORBITKV_PATH_BENCH_OUTPUT=str(output),
            )
            with (args.output / f"{pair}-{arm}.log").open("w") as log:
                subprocess.run(
                    [manifest["arms"][arm]["binary"], *manifest["command"]],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
            with output.open() as source:
                rows = list(csv.DictReader(source))
            values = {
                (row["path"], int(row["bytes"])): float(row["ns_per_operation"]) for row in rows
            }
            if len(rows) != 15 or set(values) != {(path, size) for path in PATHS for size in SIZES}:
                raise ValueError(f"incomplete or duplicated benchmark rows: {output}")
            if any(value <= 0 for value in values.values()):
                raise ValueError(f"invalid duration: {output}")
            observed[arm] = values
        for (path, size), before in observed["baseline"].items():
            after = observed["candidate"][(path, size)]
            pairs.append(
                {
                    "pair": pair,
                    "path": path,
                    "bytes": size,
                    "baseline_ns": before,
                    "candidate_ns": after,
                    "ratio": after / before,
                }
            )
    with (args.output / "pairs.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    summary = []
    for path, size in sorted({(row["path"], row["bytes"]) for row in pairs}):
        rows = [row for row in pairs if (row["path"], row["bytes"]) == (path, size)]
        ratios = [row["ratio"] for row in rows]
        summary.append(
            {
                "path": path,
                "bytes": size,
                "baseline_ns_median": statistics.median(row["baseline_ns"] for row in rows),
                "candidate_ns_median": statistics.median(row["candidate_ns"] for row in rows),
                "ratio_median": statistics.median(ratios),
                "ratio_min": min(ratios),
                "ratio_max": max(ratios),
            }
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
