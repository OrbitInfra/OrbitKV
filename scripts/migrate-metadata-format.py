#!/usr/bin/env python3
"""Dry-run or apply OrbitKV's coordinated metadata protocol cutover."""

import argparse
import base64
import json
import uuid
from pathlib import Path

import requests

ALL_NAMESPACE_PROTOCOL = "orbitkv/inventory-stream/v3"
SCOPED_PROTOCOL = "orbitkv/inventory-stream/v4"
INTEGRITY_PROTOCOL = "orbitkv/inventory-stream/v5"


def encode(value: bytes) -> str:
    return base64.b64encode(value).decode()


def prefix_end(prefix: bytes) -> bytes:
    result = bytearray(prefix)
    result[-1] += 1
    return bytes(result)


def range_prefix(endpoint: str, prefix: bytes) -> list[dict]:
    response = requests.post(
        f"{endpoint}/v3/kv/range",
        json={"key": encode(prefix), "range_end": encode(prefix_end(prefix))},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("kvs", [])


def decoded(row: dict, field: str) -> bytes:
    return base64.b64decode(row.get(field, ""))


def stream_format(protocol: str, cluster_uuid: uuid.UUID) -> bytes:
    return json.dumps(
        {"protocol": protocol, "cluster_uuid": str(cluster_uuid)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument(
        "--from-protocol",
        choices=(ALL_NAMESPACE_PROTOCOL, SCOPED_PROTOCOL),
        required=True,
    )
    parser.add_argument("--cluster-uuid", type=uuid.UUID)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    prefix = f"/orbitkv/v2/{args.cluster}/".encode()
    rows = range_prefix(args.endpoint.rstrip("/"), prefix)
    by_key = {decoded(row, "key"): row for row in rows}
    live_members = sorted(
        key.decode(errors="replace")
        for key in by_key
        if key.startswith(prefix + b"members/")
    )
    if live_members:
        raise SystemExit(f"refusing cutover with live members: {live_members}")
    format_key = prefix + b"format"
    current = by_key.get(format_key)
    if current is None:
        raise SystemExit("cluster format key is missing")
    current_value = decoded(current, "value")
    if args.cluster_uuid is None:
        raise SystemExit("--cluster-uuid is required for a reproducible cutover")
    current_protocol = args.from_protocol
    replacement_protocol = INTEGRITY_PROTOCOL
    try:
        current_stream = json.loads(current_value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        current_stream = None
    if current_stream != {
        "protocol": current_protocol,
        "cluster_uuid": str(args.cluster_uuid),
    }:
        raise SystemExit(
            f"current format is not exact {current_protocol} for --cluster-uuid"
        )
    replacement = stream_format(replacement_protocol, args.cluster_uuid)

    retired = sorted(
        key
        for key in by_key
        if key.startswith((prefix + b"blocks/", prefix + b"publishers/"))
    )
    archive = {
        "endpoint": args.endpoint,
        "cluster": args.cluster,
        "from_protocol": current_protocol,
        "to_protocol": replacement_protocol,
        "format_mod_revision": int(current["mod_revision"]),
        "current_format_base64": encode(current_value),
        "replacement_format_base64": encode(replacement),
        "retired_keys": [key.decode(errors="strict") for key in retired],
        "records": rows,
    }
    print(json.dumps({**archive, "records": f"{len(rows)} archived records"}, indent=2))
    if not args.apply:
        return
    archive_path = args.archive.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    if archive_path.is_relative_to(checkout) or archive_path.exists():
        raise SystemExit("--archive must be a new path outside the checkout")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_text(json.dumps(archive, indent=2) + "\n")

    success = [
        {
            "request_delete_range": {
                "key": encode(prefix + suffix),
                "range_end": encode(prefix_end(prefix + suffix)),
            }
        }
        for suffix in (b"blocks/", b"publishers/")
    ]
    success.append(
        {"request_put": {"key": encode(format_key), "value": encode(replacement)}}
    )
    response = requests.post(
        f"{args.endpoint.rstrip('/')}/v3/kv/txn",
        json={
            "compare": [
                {
                    "key": encode(format_key),
                    "target": "MOD",
                    "result": "EQUAL",
                    "mod_revision": str(current["mod_revision"]),
                },
                {
                    "key": encode(format_key),
                    "target": "VALUE",
                    "result": "EQUAL",
                    "value": encode(current_value),
                },
            ],
            "success": success,
        },
        timeout=30,
    )
    response.raise_for_status()
    if not response.json().get("succeeded"):
        raise SystemExit("format CAS failed; the namespace changed after the dry run")
    final = {
        decoded(row, "key"): row
        for row in range_prefix(args.endpoint.rstrip("/"), prefix)
    }
    if decoded(final[format_key], "value") != replacement:
        raise SystemExit("format verification failed")
    forbidden = [
        key.decode()
        for key in final
        if key.startswith((prefix + b"blocks/", prefix + b"publishers/"))
    ]
    if forbidden:
        raise SystemExit(f"retired metadata remains: {forbidden[:10]}")
    print(f"applied; archive={archive_path}")


if __name__ == "__main__":
    main()
