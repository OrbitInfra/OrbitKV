#!/usr/bin/env python3
"""Exercise SSD ownership and peer READ lifetimes between real Linux Managers."""

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import tempfile
import time
from contextlib import closing
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID

from ssd_persistence import Gate, frames
from tests.support.cluster import TcpGate, etcd_server, find_available_port


def member(endpoint, node):
    key = base64.b64encode(f"/orbitkv/v2/orbitkv/members/{node}".encode()).decode()
    request = Request(
        f"{endpoint}/v3/kv/range",
        data=json.dumps({"key": key}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=2) as response:
        reply = json.load(response)
    assert int(reply["header"]["revision"]) > 0
    rows = reply.get("kvs", [])
    if not rows:
        return None
    (row,) = rows
    assert row["key"] == key and int(row["lease"]) != 0
    return json.loads(base64.b64decode(row["value"]))


def partition_snapshot(shard, root):
    files = (shard, Path(str(shard) + ".index"), root / ".orbitkv-ssd-capacity")
    return {
        file.name: [file.stat().st_ino, hashlib.sha256(file.read_bytes()).hexdigest()]
        for file in files
    }


def control_bind_failure(gate, path, endpoint):
    node = "bind-contender"
    ttl = 300
    cluster = (
        "--node-id",
        node,
        "--etcd-endpoints",
        endpoint,
        "--membership-ttl-secs",
        str(ttl),
    )
    path.mkdir()
    storage = {"cluster_args": cluster, "shared_namespace": 0, "root_capacity": 4096}
    started = time.monotonic()
    with socket.create_server(("127.0.0.1", 0)) as occupied:
        failed = gate.manager(
            path,
            **storage,
            address=f"127.0.0.1:{occupied.getsockname()[1]}",
            pauses=("ssd_root_admitted",),
            defer_start=True,
        )
        failed.reached("ssd_root_admitted")
        registered = member(endpoint, node)
        assert registered["owner"]["endpoint"] == failed.address
        failed.release("ssd_root_admitted")
        failed.rejected("AddrInUse")
        assert member(endpoint, node) is None, (
            "failed gRPC listener retained its membership"
        )
        assert "SSD cache initialized at" in failed.log_path.read_text()
        failed.close()

        port = find_available_port()
        retry = gate.manager(path, **storage, address=f"127.0.0.1:{port}")

    def listening():
        with socket.socket() as probe:
            probe.settimeout(1)
            return probe.connect_ex(("127.0.0.1", port)) == 0

    retry.wait(listening)
    replacement = member(endpoint, node)
    assert replacement["owner"]["endpoint"] == retry.address
    assert replacement["epoch"] > registered["epoch"]
    assert replacement["owner"]["incarnation"] != registered["owner"]["incarnation"]
    elapsed = time.monotonic() - started
    assert elapsed < ttl, "retry must precede natural expiry of the first lease"
    retry.close()
    assert retry.process.returncode == 0
    assert member(endpoint, node) is None
    (gate.artifacts / "control-bind-failure.json").write_text(
        json.dumps(
            {
                "failed": registered,
                "retry": replacement,
                "ttl_seconds": ttl,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )


def cluster_controls(gate, endpoint, credentials, token):
    def node(name, data, *, address=None, advertise=None, **options):
        peer_args = ("--peer-advertise-addr", advertise) if advertise else ()
        return gate.manager(
            data,
            address=address or f"127.0.0.1:{find_available_port()}",
            cluster_args=(
                "--node-id",
                name,
                "--etcd-endpoints",
                endpoint,
                "--membership-ttl-secs",
                "12",
                "--router-credentials",
                str(credentials),
                *peer_args,
            ),
            **options,
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

    return node, router


def cold_placements(observer, router, hashes, expected):
    def current():
        result = router(observer, "lookup", {"hashes": hashes})
        assert not result["local"], "requester must remain cold before its peer READ"
        rows = result["placements"]
        assert [row["hash"] for row in rows] == hashes
        owners = [[p["owner"] for p in row["peers"]] for row in rows]
        if owners == expected and all(
            row["peer_coverage"] == "complete_at_watermarks" for row in rows
        ):
            return rows
        return None

    return observer.wait(current)


def handoff(gate, path, endpoint, node, router, *, expire=False):
    storage = {}
    shard = path
    if expire:
        path.mkdir()
        storage = {"shared_namespace": 0, "capacity": 4096, "root_capacity": 4096}
        shard = path / f"orbitkv-ssd-orbitkv:v2:{0:064x}" / "shard-000000.dat"
    source = node("writer", path, **storage)
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
        return cold_placements(observer, router, hashes, expected)

    rows = placements([[old], [old]])
    old_sequence = rows[0]["peers"][0]["sequence"]
    grant = {"source": old, "requester": requester, "residency": old_sequence}

    def authorize(probe):
        return observer.call("grant", id=good_id, grant=probe)

    issued = authorize(grant)
    assert "window" in issued, issued
    UUID(issued["window"])
    assert issued["epoch"]
    expiry = {}
    if expire:
        writer = member(endpoint, "writer")
        assert writer["owner"] == old
        source.process.send_signal(signal.SIGSTOP)

        def stopped():
            pid, status = os.waitpid(source.process.pid, os.WUNTRACED | os.WNOHANG)
            return status if pid == source.process.pid else None

        status = source.wait(stopped)
        assert os.WIFSTOPPED(status) and os.WSTOPSIG(status) == signal.SIGSTOP
        observer.wait(lambda: member(endpoint, "writer") is None)
        assert member(endpoint, "requester")["owner"]["incarnation"] == requester
        placements([[], []])
        original = partition_snapshot(shard, path)
        challenger = node(
            "contender",
            path,
            **storage,
            pauses=("ssd_root_admitted",),
            defer_start=True,
        )
        challenger.reached("ssd_root_admitted")
        registered = member(endpoint, "contender")
        assert registered["owner"]["endpoint"] == challenger.address != source.address
        challenger.release("ssd_root_admitted")
        challenger.rejected("SSD shard is already owned")
        challenger.close()
        # Engine construction failed after membership registration. The process
        # must revoke its own lease before returning the storage error.
        assert member(endpoint, "contender") is None
        gate.manager(
            path,
            shared_namespace=1,
            capacity=4096,
            reject_with="shared SSD root capacity exceeded",
        ).close()
        assert partition_snapshot(shard, path) == original

        # A resumed Manager keeps local storage but its expired incarnation
        # cannot regain distributed authority.
        source.process.send_signal(signal.SIGCONT)

        def fenced():
            with urlopen(f"http://{source.http}/cache/metadata", timeout=2) as response:
                return json.load(response)["index"]["registration_valid"] is False

        source.wait(fenced)
        assert member(endpoint, "writer") is None
        denied = authorize(grant)
        assert denied["grpc_code"] == "FailedPrecondition", denied
        assert denied["detail"] == "stale owner or residency candidate", denied
        source.read(good_id)
        with shard.open("rb") as reader:
            source.close(kill=True)
            node(
                "contender",
                path,
                **storage,
                reject_with="drained prior file users",
            ).close()
            assert member(endpoint, "contender") is None
            assert os.fstat(reader.fileno()).st_ino == original[shard.name][0]
            assert partition_snapshot(shard, path) == original
        expiry = {
            "expired_epoch": writer["epoch"],
            "rejected_contender": registered["owner"],
            "resumed_grant_status": denied["grpc_code"],
            "stopped_writer_rejection": "SSD shard is already owned",
            "other_namespace_rejection": "shared SSD root capacity exceeded",
            "held_reader_rejection": "drained prior file users",
            "preserved_files": original,
        }
    else:
        source.close(kill=True)
    placements([[], []])

    commits = frames(Path(str(shard) + ".index"))
    (bad,) = [
        item
        for item in commits
        if item.get("checksum") is not None and item["key"]["hash"] == hashes[1]
    ]
    with shard.open("r+b", buffering=0) as data:
        data.seek(bad["offset"])
        original = data.read(1)
        data.seek(bad["offset"])
        data.write(bytes([original[0] ^ 1]))
        os.fsync(data.fileno())

    replacement = node(
        "writer" if expire else "replacement",
        path,
        **storage,
        address=old["endpoint"],
        pauses=("ssd_before_verification",),
    )
    replacement.reached("ssd_before_verification")
    reopened = lookup(replacement)
    new = reopened["source"]
    assert (
        new["endpoint"] == old["endpoint"] and new["incarnation"] != old["incarnation"]
    )
    if expire:
        successor = member(endpoint, "writer")
        assert (
            successor["owner"] == new and successor["epoch"] > expiry["expired_epoch"]
        )
        expiry["replacement_epoch"] = successor["epoch"]
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
    receipt = "expired-owner-handoff.json" if expire else "handoff.json"
    (gate.artifacts / receipt).write_text(
        json.dumps(
            {
                **expiry,
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


def transfer_metrics(manager, *names):
    values = {}
    with urlopen(f"http://{manager.http}/metrics", timeout=2) as response:
        for line in response.read().decode().splitlines():
            for name in names:
                if line.startswith((name + "{", name + " ")):
                    assert name not in values, line
                    values[name] = float(line.split()[-1])
    return values


def transfer_charges(source, requester):
    values = transfer_metrics(
        source, "orbitkv_transfer_lock_active", "orbitkv_transfer_reserved_bytes"
    )
    values.update(
        transfer_metrics(requester, "orbitkv_transfer_completion_outstanding")
    )
    assert len(values) == 3, values
    return values


def live_reclaim_authorized_read(gate, root, endpoint, node, router):
    root.mkdir()
    capacity = 4096
    donor = node("live-donor", root, shared_namespace=0, root_capacity=capacity)
    observer = node("live-reader", root.parent / "reader")
    key = 1
    stored = donor.call("write", id=key)
    assert stored["present"]
    hashes = [stored["hash"]]
    original = donor.read(key)["slots"]
    registered = member(endpoint, "live-donor")
    source = registered["owner"]
    requester = router(observer, "lookup", {"hashes": hashes})["source"]["incarnation"]
    (row,) = cold_placements(observer, router, hashes, [[source]])
    (residency,) = row["peers"]
    assert residency["metadata"]["medium"] == "Ssd"
    idle = transfer_charges(donor, observer)
    assert all(value == 0 for value in idle.values()), idle
    gate.manager(
        root, shared_namespace=1, reject_with="shared SSD root capacity exceeded"
    ).close()

    point = "peer_read_authorized"
    observer.pause(point)
    observer.send("peer_read", id=key)
    observer.reached(point)
    try:
        held = transfer_charges(donor, observer)
        assert held["orbitkv_transfer_lock_active"] == 1
        assert held["orbitkv_transfer_completion_outstanding"] == 1
        assert held["orbitkv_transfer_reserved_bytes"] >= sum(
            len(slot["bytes"]) for slot in original
        )
        namespace = "orbitkv:v2:" + "0" * 64
        reclaimed = subprocess.run(
            [
                str(gate.binary),
                "reclaim-ssd-partition",
                "--root",
                str(root),
                "--namespace",
                namespace,
                "--wait-for-owner-secs",
                "30",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=40,
        )
        assert (
            reclaimed.stdout.strip()
            == f"Reclaimed {capacity} payload bytes from {namespace}"
        )
        assert not donor.call("report")["state"]["ssd_attached"]
        cold_placements(observer, router, hashes, [[]])
        assert member(endpoint, "live-donor") == registered
        newcomer = gate.manager(root, shared_namespace=1)
        overwrite = newcomer.call("write", namespace=1, id=key, fill=173)
        assert overwrite["present"] and overwrite["hash"] == stored["hash"]
        reused = newcomer.read(key, namespace=1, fill=173)["slots"]
        assert reused != original
        # A new window tests the old residency without reusing the admitted ticket.
        denied = donor.call(
            "grant",
            id=key,
            grant={
                "source": source,
                "requester": requester,
                "residency": residency["sequence"],
            },
        )
        assert denied["grpc_code"] == "FailedPrecondition", denied
        assert denied["detail"] == "stale owner or residency candidate", denied
        after_reuse = transfer_charges(donor, observer)
        assert after_reuse == held, after_reuse
    finally:
        observer.release(point)

    received = observer.reply()
    assert received["present"] and received["slots"] == original

    def released():
        current = transfer_charges(donor, observer)
        return current if current == idle else None

    drained = donor.wait(released)
    (gate.artifacts / "live-reclaim-authorized-read.json").write_text(
        json.dumps(
            {
                "source": source,
                "residency": residency["sequence"],
                "reclaimed_bytes": capacity,
                "stale_grant_status": denied["grpc_code"],
                "verified_peer_slot_bytes": [
                    len(slot["bytes"]) for slot in received["slots"]
                ],
                "peer_slot_sha256": [
                    hashlib.sha256(bytes(s["bytes"])).hexdigest()
                    for s in received["slots"]
                ],
                "replacement_slot_sha256": [
                    hashlib.sha256(bytes(s["bytes"])).hexdigest() for s in reused
                ],
                "held": held,
                "after_reuse": after_reuse,
                "drained": drained,
            },
            indent=2,
        )
    )
    observer.close()
    newcomer.close()
    donor.close()


def source_releases(source):
    releases = []
    for line in source.log_path.read_text().splitlines():
        if "Peer release handled:" in line:
            match = re.search(
                r"Peer release handled: ticket=TransferTicket \{ window: ([0-9a-f-]+), "
                r"slot: (\d+), generation: (\d+) \} released_blocks=(\d+)",
                line,
            )
            assert match, line
            window, slot, generation, blocks = match.groups()
            releases.append(
                {
                    "ticket": {
                        "window": str(UUID(window)),
                        "slot": int(slot),
                        "generation": int(generation),
                    },
                    "released_blocks": int(blocks),
                }
            )
    return releases


def release_during_grpc_outage(gate, path, endpoint, node, router, *, lose_reply=False):
    # Reserve the bind address until the proxy has chosen its own listener.
    with socket.socket() as reserved:
        port = find_available_port()
        reserved.bind(("127.0.0.1", port))
        address = f"127.0.0.1:{port}"
        proxy = TcpGate(f"http://{address}")
    with closing(proxy):
        advertised = proxy.endpoint.removeprefix("http://")
        source = node("release-source", path, address=address, advertise=advertised)
        point = "peer_read_authorized"
        reader = node("release-reader", path.parent / "reader", pauses=(point,))

        def members():
            return {
                name: member(endpoint, name)
                for name in ("release-source", "release-reader")
            }

        registered = members()
        owner = registered["release-source"]["owner"]
        assert (
            owner["endpoint"] == advertised
            and reader.address == registered["release-reader"]["owner"]["endpoint"]
        )
        stored = source.call("write", id=1)
        assert stored["present"]
        original = source.read(1)["slots"]
        payload_bytes = sum(len(slot["bytes"]) for slot in original)
        (row,) = cold_placements(reader, router, [stored["hash"]], [[owner]])
        assert row["peers"][0]["metadata"]["medium"] == "Ssd"

        idle = transfer_charges(source, reader)
        assert all(value == 0 for value in idle.values()), idle
        retries = "orbitkv_transfer_completion_retries_total"
        fetched = "orbitkv_remote_fetch_bytes_total"
        reader.send("peer_read", id=1)
        reader.reached(point)
        held = transfer_charges(source, reader)
        assert held["orbitkv_transfer_lock_active"] == 1
        assert held["orbitkv_transfer_reserved_bytes"] >= payload_bytes
        assert held["orbitkv_transfer_completion_outstanding"] == 1
        # Counters on a fresh requester can be absent until their first event.
        before = transfer_metrics(reader, retries, fetched)
        assert all(value == 0 for value in before.values()), before

        receipt = {}
        if lose_reply:
            reply_point = "peer_release_before_reply"
            source.pause(reply_point)
            reader.release(point)
            source.reached(reply_point)
            proxy.drop_responses()
            unconfirmed = dict(idle, orbitkv_transfer_completion_outstanding=1)
            before_reply = transfer_charges(source, reader)
            assert before_reply == unconfirmed, before_reply
            (first,) = source_releases(source)
            assert first["released_blocks"] == 1, first
            receipt["released_before_reply"] = before_reply
            source.release(reply_point)
            dropped = proxy.wait_for_dropped_response()
            assert dropped > 0, dropped
            receipt["dropped_control_response_bytes"] = dropped
        else:
            proxy.partition()
            reader.release(point)
        received = reader.reply()
        assert received["present"] and received["slots"] == original

        def release_failed():
            counts = transfer_metrics(reader, retries, fetched)
            return counts if counts.get(retries, 0) > 0 else None

        blocked = reader.wait(release_failed)
        assert blocked[fetched] == payload_bytes, blocked
        if lose_reply:
            # Discard the HTTP/2 connections that lost bytes before reconnecting.
            proxy.partition()
            before_reconnect = source_releases(source)
            receipt["source_releases_before_reconnect"] = before_reconnect
        retained = transfer_charges(source, reader)
        assert retained == (unconfirmed if lose_reply else held), retained
        assert members() == registered

        proxy.heal()

        def released():
            charges = transfer_charges(source, reader)
            return charges if charges == idle else None

        drained = reader.wait(released)
        after = transfer_metrics(reader, retries, fetched)
        assert after[fetched] == blocked[fetched] and after[retries] >= blocked[retries]
        assert members() == registered
        if lose_reply:
            releases = source_releases(source)
            assert releases[: len(before_reconnect)] == before_reconnect
            recovered = releases[len(before_reconnect) :]
            assert recovered, releases
            for duplicate in releases[1:]:
                assert duplicate["ticket"] == first["ticket"], releases
                assert duplicate["released_blocks"] == 0, releases
            receipt["source_releases_after_reconnect"] = recovered
        name = "grpc-release-ack-loss" if lose_reply else "grpc-release-retry"
        (gate.artifacts / f"{name}.json").write_text(
            json.dumps(
                {
                    **receipt,
                    "members": registered,
                    "held": held,
                    "read_complete_while_disconnected": blocked,
                    "retained": retained,
                    "drained": drained,
                    "after_reconnect": after,
                    "verified_slot_bytes": [len(s["bytes"]) for s in received["slots"]],
                    "slot_sha256": [
                        hashlib.sha256(bytes(s["bytes"])).hexdigest()
                        for s in received["slots"]
                    ],
                },
                indent=2,
            )
        )
        reader.close()
        source.close()


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
            node, router = cluster_controls(gate, endpoint, config, token)
            gate.case(
                "owner-handoff",
                lambda path: handoff(gate, path, endpoint, node, router),
            )
            gate.case(
                "expired-owner-handoff",
                lambda path: handoff(gate, path, endpoint, node, router, expire=True),
            )
            gate.case(
                "control-bind-failure",
                lambda path: control_bind_failure(gate, path, endpoint),
            )
            gate.case(
                "live-reclaim-authorized-read",
                lambda path: live_reclaim_authorized_read(
                    gate, path, endpoint, node, router
                ),
            )
            gate.case(
                "grpc-release-retry",
                lambda path: release_during_grpc_outage(
                    gate, path, endpoint, node, router
                ),
            )
            gate.case(
                "grpc-release-ack-loss",
                lambda path: release_during_grpc_outage(
                    gate, path, endpoint, node, router, lose_reply=True
                ),
            )


if __name__ == "__main__":
    main()
