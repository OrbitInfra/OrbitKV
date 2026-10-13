"""Static fail-closed checks for a frozen three-host etcd campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

try:
    from .environment import (
        PROXY_ENVIRONMENT_VARIABLES,
        validate_network_environment_policy,
        validated_manager_socket_paths,
    )
    from .integrity import verify_inputs
except ImportError:
    from environment import (
        PROXY_ENVIRONMENT_VARIABLES,
        validate_network_environment_policy,
        validated_manager_socket_paths,
    )
    from integrity import verify_inputs
PHASES = [
    "bootstrap",
    "orbitkv_consumer",
    "follower_stop_rejoin",
    "leader_stop_rejoin",
    "quorum_loss",
    "quorum_restore",
    "lease_revoke_replacement",
    "cold_observer_final_oracle",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root: Path, approved_input_manifest_sha256: str) -> dict:
    inputs = verify_inputs(root, approved_input_manifest_sha256)
    settings = json.loads((root / "inputs/settings.json").read_text())
    if settings["formal_launch_authorized"] is not False:
        raise RuntimeError(
            "Frozen three-host contract failed: settings['formal_launch_authorized'] is False"
        )
    if not settings["inventory_protocol"] == "orbitkv/inventory-stream/v4":
        raise RuntimeError(
            "Frozen three-host contract failed: settings['inventory_protocol'] == 'orbitkv/inventory-stream/v4'"
        )
    validate_network_environment_policy(settings["network_environment"])
    fake_proxy = urlsplit(settings["proxy_readiness"]["fake_proxy_url"])
    if not fake_proxy.scheme == "http":
        raise RuntimeError("Frozen three-host contract failed: fake_proxy.scheme == 'http'")
    if not (fake_proxy.hostname is not None and fake_proxy.port is not None):
        raise RuntimeError(
            "Frozen three-host contract failed: fake_proxy.hostname is not None and fake_proxy.port is not None"
        )
    if fake_proxy.hostname not in {"127.0.0.1", "::1"}:
        raise RuntimeError(
            "Frozen three-host contract failed: fake_proxy.hostname in {'127.0.0.1', '::1'}"
        )
    nodes = settings["nodes"]
    if not len(nodes) == 3:
        raise RuntimeError("Frozen three-host contract failed: len(nodes) == 3")
    if not len({node["name"] for node in nodes}) == 3:
        raise RuntimeError(
            "Frozen three-host contract failed: len({node['name'] for node in nodes}) == 3"
        )
    if not len({node["hostname"] for node in nodes}) == 3:
        raise RuntimeError(
            "Frozen three-host contract failed: len({node['hostname'] for node in nodes}) == 3"
        )
    if not len({node["boot_id"] for node in nodes}) == 3:
        raise RuntimeError(
            "Frozen three-host contract failed: len({node['boot_id'] for node in nodes}) == 3"
        )
    if not len({node["ipv6"] for node in nodes}) == 3:
        raise RuntimeError(
            "Frozen three-host contract failed: len({node['ipv6'] for node in nodes}) == 3"
        )
    for node in nodes:
        client = urlsplit(node["client_url"])
        peer = urlsplit(node["peer_url"])
        if not client.scheme == peer.scheme == "http":
            raise RuntimeError(
                "Frozen three-host contract failed: client.scheme == peer.scheme == 'http'"
            )
        if not client.hostname == peer.hostname == node["ipv6"]:
            raise RuntimeError(
                "Frozen three-host contract failed: client.hostname == peer.hostname == node['ipv6']"
            )
        if not client.port == settings["etcd"]["client_port"]:
            raise RuntimeError(
                "Frozen three-host contract failed: client.port == settings['etcd']['client_port']"
            )
        if not peer.port == settings["etcd"]["peer_port"]:
            raise RuntimeError(
                "Frozen three-host contract failed: peer.port == settings['etcd']['peer_port']"
            )
    if not settings["etcd"]["sha256"] == sha256(root / "frozen/etcd"):
        raise RuntimeError(
            "Frozen three-host contract failed: settings['etcd']['sha256'] == sha256(root / 'frozen/etcd')"
        )
    runtime = settings["orbitkv_runtime"]
    source = Path(runtime["local_evidence_root"])
    if not runtime["manager_sha256"] == sha256(source / "frozen/bin/orbitkv-cache-manager"):
        raise RuntimeError(
            "Frozen three-host contract failed: runtime['manager_sha256'] == sha256(source / 'frozen/bin/orbitkv-cache-manager')"
        )
    if not runtime["server_tests_sha256"] == sha256(source / "frozen/bin/server-tests"):
        raise RuntimeError(
            "Frozen three-host contract failed: runtime['server_tests_sha256'] == sha256(source / 'frozen/bin/server-tests')"
        )
    if not runtime["extension_sha256"] == sha256(
        source / "frozen/wheel-members/orbitkv.cpython-311-x86_64-linux-gnu.so"
    ):
        raise RuntimeError(
            "Frozen three-host contract failed: runtime['extension_sha256'] == sha256(source / 'frozen/wheel-members/orbitkv.cpython-311-x86_64-linux-gnu.so')"
        )
    if not runtime["tent_sha256"] == sha256(source / "frozen/native/libtent_shared.so"):
        raise RuntimeError(
            "Frozen three-host contract failed: runtime['tent_sha256'] == sha256(source / 'frozen/native/libtent_shared.so')"
        )
    if not runtime["projection_manifest_sha256"] == sha256(source / "projection/SHA256SUMS"):
        raise RuntimeError(
            "Frozen three-host contract failed: runtime['projection_manifest_sha256'] == sha256(source / 'projection/SHA256SUMS')"
        )
    if (
        not runtime["pressure_rounds"] * runtime["pressure_cadence_ms"] / 1000
        < settings["budgets"]["phase_seconds"]
    ):
        raise RuntimeError(
            "Frozen three-host contract failed: runtime['pressure_rounds'] * runtime['pressure_cadence_ms'] / 1000 < settings['budgets']['phase_seconds']"
        )
    runtime_ports = {
        runtime["manager_peer_port"],
        runtime["manager_http_port"],
        runtime["observer_peer_port"],
        runtime["observer_http_port"],
        runtime["query_source_peer_port"],
        runtime["query_source_http_port"],
    }
    if not len(runtime_ports) == 6:
        raise RuntimeError("Frozen three-host contract failed: len(runtime_ports) == 6")
    if not all(0 < port <= 65535 for port in runtime_ports):
        raise RuntimeError(
            "Frozen three-host contract failed: all((0 < port <= 65535 for port in runtime_ports))"
        )
    socket_paths = validated_manager_socket_paths(runtime)
    if not len(socket_paths) == 3:
        raise RuntimeError("Frozen three-host contract failed: len(socket_paths) == 3")
    if not settings["budgets"] == {
        "cleanup_seconds": 180,
        "command_seconds": 30,
        "global_seconds": 2400,
        "listener_preflight_seconds": 300,
        "normal_stop_seconds": 30,
        "phase_seconds": 240,
    }:
        raise RuntimeError(
            "Frozen three-host contract failed: settings['budgets'] == {'cleanup_seconds': 180, 'command_seconds': 30, 'global_seconds': 2400, 'listener_preflight_seconds': 300, 'normal_stop_seconds': 30, 'phase_seconds': 240}"
        )
    if not settings["resource_lock"]["path"].startswith("/tmp/"):
        raise RuntimeError(
            "Frozen three-host contract failed: settings['resource_lock']['path'].startswith('/tmp/')"
        )
    if not settings["resource_lock"]["resources"]:
        raise RuntimeError(
            "Frozen three-host contract failed: settings['resource_lock']['resources']"
        )
    if f"uds-root-{Path(runtime['uds_root']).name}" not in settings["resource_lock"]["resources"]:
        raise RuntimeError(
            "Frozen three-host contract failed: f\"uds-root-{Path(runtime['uds_root']).name}\" in settings['resource_lock']['resources']"
        )
    if not settings["resource_gate"]["external_owners"]:
        raise RuntimeError(
            "Frozen three-host contract failed: settings['resource_gate']['external_owners']"
        )
    if not settings["resource_gate"]["process_patterns"]:
        raise RuntimeError(
            "Frozen three-host contract failed: settings['resource_gate']['process_patterns']"
        )
    for owner in settings["resource_gate"]["external_owners"]:
        if owner["node"] not in {node["name"] for node in nodes}:
            raise RuntimeError(
                "Frozen three-host contract failed: owner['node'] in {node['name'] for node in nodes}"
            )
        if not Path(owner["status_path"]).is_absolute():
            raise RuntimeError(
                "Frozen three-host contract failed: Path(owner['status_path']).is_absolute()"
            )
        if not owner["terminal_states"]:
            raise RuntimeError("Frozen three-host contract failed: owner['terminal_states']")
        if not ("pid" not in owner and "start_ticks" not in owner):
            raise RuntimeError(
                "Frozen three-host contract failed: 'pid' not in owner and 'start_ticks' not in owner"
            )
    if not settings["stop_rules"][0].startswith("one serial campaign"):
        raise RuntimeError(
            "Frozen three-host contract failed: settings['stop_rules'][0].startswith('one serial campaign')"
        )
    if not settings["network_environment"]["unset"] == list(PROXY_ENVIRONMENT_VARIABLES):
        raise RuntimeError(
            "Frozen three-host contract failed: settings['network_environment']['unset'] == list(PROXY_ENVIRONMENT_VARIABLES)"
        )
    return {
        "status": "passed",
        "nodes": 3,
        "phases": PHASES,
        "input_manifest_sha256": approved_input_manifest_sha256,
        "input_files": sorted(inputs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--approved-input-manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.approved_input_manifest_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
