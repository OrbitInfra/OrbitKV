#!/usr/bin/env python3
"""Exercise an exclusive SSD handoff between real Managers on the same Linux host."""

import argparse
import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID

from ssd_persistence import Gate, frames
from tests.support.cluster import etcd_server


def handoff(gate, path, endpoint, credentials, token):
    def node(name, data, *, address=None, pauses=()):
        return gate.manager(
            data,
            address=address,
            pauses=pauses,
            cluster_args=(
                "--node-id",
                name,
                "--etcd-endpoints",
                endpoint,
                "--membership-ttl-secs",
                "12",
                "--router-credentials",
                str(credentials),
            ),
        )

    def router(manager, route, body, expected=200):
        req = Request(
            f"http://{manager.http}/v1/router/{route}",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )

        def receive():
            try:
                with urlopen(req, timeout=2) as response:
                    assert response.status == expected
                    return json.load(response)
            except HTTPError as error:
                if error.code == 503:
                    return None
                if error.code != expected:
                    raise
                return error.read().decode()
            except URLError as error:
                if isinstance(error.reason, ConnectionRefusedError):
                    return None
                raise

        return manager.wait(receive)

    source = node("writer", path)
    observer = node("requester", path.parent / "requester")
    good_id = 1
    good = source.call("write", id=good_id)
    damaged = source.call("write", id=2)
    assert good["present"] and damaged["present"]
    hashes = [good["hash"], damaged["hash"]]

    def lookup(manager):
        return router(manager, "lookup", {"hashes": hashes})

    initial = lookup(source)
    old = initial["source"]
    assert [r["hash"] for r in initial["local"]] == hashes
    requester = lookup(observer)["source"]["incarnation"]

    def placements(expected):
        def current():
            result = lookup(observer)
            assert not result["local"], (
                "requester must remain cold until its final peer READ"
            )
            rows = result["placements"]
            assert [row["hash"] for row in rows] == hashes
            owners = [[p["owner"] for p in row["peers"]] for row in rows]
            if owners == expected and all(
                row["peer_coverage"] == "complete_at_watermarks" for row in rows
            ):
                return rows
            return None

        return observer.wait(current)

    rows = placements([[old], [old]])
    old_sequence = rows[0]["peers"][0]["sequence"]
    grant = {"source": old, "requester": requester, "residency": old_sequence}

    def authorize(probe):
        return observer.call("grant", id=good_id, grant=probe)

    issued = authorize(grant)
    assert "window" in issued, issued
    UUID(issued["window"])
    assert issued["epoch"]
    source.close(kill=True)
    placements([[], []])

    commits = frames(Path(str(path) + ".index"))
    (bad,) = [
        item
        for item in commits
        if item.get("checksum") is not None and item["key"]["hash"] == hashes[1]
    ]
    with path.open("r+b", buffering=0) as data:
        data.seek(bad["offset"])
        original = data.read(1)
        data.seek(bad["offset"])
        data.write(bytes([original[0] ^ 1]))
        os.fsync(data.fileno())

    replacement = node(
        "replacement",
        path,
        address=old["endpoint"],
        pauses=("ssd_before_verification",),
    )
    replacement.reached("ssd_before_verification")
    reopened = lookup(replacement)
    new = reopened["source"]
    assert (
        new["endpoint"] == old["endpoint"] and new["incarnation"] != old["incarnation"]
    )
    assert reopened["local"] == []
    assert (
        router(replacement, "events", {"cursor": initial["cursor"]}, 409)
        == "resync_required"
    )

    def new_view_installed():
        with urlopen(
            f"http://{observer.http}/cache/metadata/owners", timeout=2
        ) as response:
            return any(
                v["owner"] == new["incarnation"] and v["fresh"]
                for v in json.load(response)
            )

    observer.wait(new_view_installed)
    placements([[], []])
    observer.read(good_id, present=False, operation="peer_read")
    replacement.release("ssd_before_verification")
    replacement.recovered()
    recovery = replacement.call("report")["state"]["recovery"]
    assert recovery["drops"].get("data_checksum") == 1, recovery
    published = lookup(replacement)
    assert [row["hash"] for row in published["local"]] == [hashes[0]]
    rows = placements([[new], []])
    current = rows[0]["peers"][0]
    assert current["metadata"]["medium"] == "Ssd"
    assert current["sequence"] == published["local"][0]["sequence"]

    old_ticket = {**grant, "window": issued["window"]}
    stale = authorize(old_ticket)
    assert stale["grpc_code"] == "FailedPrecondition", stale
    # Use the verified residency for both probes so only the owner changes.
    current_residency = {**old_ticket, "residency": current["sequence"]}
    retired = authorize(current_residency)
    assert retired["grpc_code"] == "FailedPrecondition", retired
    rebound = authorize({**current_residency, "source": new})
    assert rebound["grpc_code"] == "NotFound", rebound
    observer.read(2, present=False, operation="peer_read")
    received = observer.read(good_id, operation="peer_read")
    (gate.artifacts / "handoff.json").write_text(
        json.dumps(
            {
                "old_owner": old,
                "new_owner": new,
                "old_ticket_status": stale["grpc_code"],
                "retired_owner_status": retired["grpc_code"],
                "rebound_ticket_status": rebound["grpc_code"],
                "checksum_drops": recovery["drops"]["data_checksum"],
                "verified_peer_slot_bytes": [
                    len(slot["bytes"]) for slot in received["slots"]
                ],
            },
            indent=2,
        )
    )
    observer.close()
    replacement.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["binary", "root", "artifacts"]:
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("ORBITKV_REQUIRE_ENGINE_SSD", "ORBITE_REQUIRE_IO_URING"):
        if os.environ.get(name) != "1":
            parser.error(f"{name}=1 is required")
    if not os.environ.get("ETCD_BIN"):
        parser.error("ETCD_BIN is required")
    gate = Gate(args)
    with tempfile.TemporaryDirectory(prefix="okv-handoff-") as temporary:
        directory = Path(temporary)
        token = secrets.token_hex(32)
        config = directory / "credentials.json"
        config.write_text(
            json.dumps(
                [
                    {
                        "namespace": "orbitkv:v2:" + "0" * 64,
                        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    }
                ]
            )
        )
        with etcd_server(directory) as (endpoint, _):
            gate.case(
                "owner-handoff",
                lambda path: handoff(gate, path, endpoint, config, token),
            )


if __name__ == "__main__":
    main()
