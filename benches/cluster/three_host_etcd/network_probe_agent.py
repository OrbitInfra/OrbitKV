"""Hold exact etcd ports for bounded directed IPv6 connectivity probes."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import socket
import time
from pathlib import Path

try:
    from benches.cluster.three_host_etcd.environment import direct_network_environment_evidence
    from benches.cluster.three_host_etcd.integrity import verify_inputs, verify_process_runtime
except ImportError:
    from environment import direct_network_environment_evidence
    from integrity import verify_inputs, verify_process_runtime


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--approved-input-manifest-sha256", required=True)
    args = parser.parse_args()
    verify_inputs(args.root, args.approved_input_manifest_sha256)
    settings = json.loads((args.root / "inputs/settings.json").read_text())
    runtime_evidence = verify_process_runtime(args.root, settings, args.node)
    direct_network_environment_evidence(os.environ, settings["network_environment"])
    node = next(item for item in settings["nodes"] if item["name"] == args.node)
    if socket.gethostname() != node["hostname"]:
        raise RuntimeError("network probe hostname mismatch")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if boot_id != node["boot_id"]:
        raise RuntimeError("network probe boot ID mismatch")

    ready = args.root / f"{args.node}.network-probe.json"
    stop = args.root / f"{args.node}.network-probe.stop"
    result_path = args.root / f"{args.node}.network-probe.result.json"
    if ready.exists() or stop.exists() or result_path.exists():
        raise FileExistsError("network probe evidence already exists")
    ports = (settings["etcd"]["client_port"], settings["etcd"]["peer_port"])
    selector = selectors.DefaultSelector()
    listeners = []
    for port in ports:
        listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        listener.bind((node["ipv6"], port))
        listener.listen()
        listener.setblocking(False)
        selector.register(listener, selectors.EVENT_READ, data=port)
        listeners.append(listener)
    record = {
        "status": "listening",
        "process_runtime": runtime_evidence,
        "node": args.node,
        "hostname": socket.gethostname(),
        "boot_id": boot_id,
        "pid": os.getpid(),
        "start_ticks": start_ticks(os.getpid()),
        "ipv6": node["ipv6"],
        "ports": list(ports),
        "connections": [],
        "started_unix_ns": time.time_ns(),
    }
    atomic_json(ready, record)
    deadline = time.monotonic() + settings["budgets"]["listener_preflight_seconds"]
    try:
        while not stop.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("network probe lifetime expired")
            for key, _ in selector.select(timeout=0.1):
                connection, address = key.fileobj.accept()
                with connection:
                    connection.settimeout(2)
                    payload = connection.recv(1)
                record["connections"].append(
                    {
                        "port": key.data,
                        "source_ipv6": address[0],
                        "payload_hex": payload.hex(),
                        "observed_unix_ns": time.time_ns(),
                    }
                )
                atomic_json(ready, record)
    finally:
        for listener in listeners:
            selector.unregister(listener)
            listener.close()
        selector.close()
    atomic_json(
        result_path,
        {**record, "status": "stopped", "finished_unix_ns": time.time_ns()},
    )


if __name__ == "__main__":
    main()
