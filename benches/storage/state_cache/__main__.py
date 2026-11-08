"""python -m benches.storage.state_cache --help"""

import argparse
import json
from pathlib import Path

from .campaign import campaign
from .schema import MAX_CONFIG_BYTES, load_config
from .sources import SOURCES


def main():
    parser = argparse.ArgumentParser(
        description="Design-v2 simulator; no hardware evidence or dataset import"
    )
    parser.add_argument("--model", required=True, choices=sorted(SOURCES))
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="external official config.json, checked against pin/hash",
    )
    parser.add_argument(
        "--parameters",
        required=True,
        type=Path,
        help="external JSON of explicit sizes, costs, capacities and sweep",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="new external artifact directory"
    )
    parser.add_argument("--max-runs", type=int, default=512)
    parser.add_argument(
        "--plot", action="store_true", help="requires matplotlib; writes plot outside Git"
    )
    args = parser.parse_args()
    schema = load_config(args.model, args.config)
    with args.parameters.open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    if len(raw) > MAX_CONFIG_BYTES:
        parser.error("parameter file exceeds 1 MiB")
    rows = campaign(schema, json.loads(raw), args.output, maximum=args.max_runs, plots=args.plot)
    failures = sum(row["status"] != "ok" for row in rows)
    print(f"simulation only: {len(rows)} runs, {failures} failed; artifacts: {args.output}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
