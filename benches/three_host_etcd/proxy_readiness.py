"""Run the bounded three-host direct-network and etcd bootstrap readiness gate."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import requests

try:
    from .campaign import Campaign, atomic_json
    from .environment import (
        PROXY_ENVIRONMENT_VARIABLES,
        direct_network_environment_evidence,
    )
except ImportError:
    from campaign import Campaign, atomic_json
    from environment import PROXY_ENVIRONMENT_VARIABLES, direct_network_environment_evidence


def _cleanup(campaign: Campaign) -> dict:
    campaign.cleanup_deadline = time.monotonic() + campaign.settings["budgets"]["cleanup_seconds"]
    cleanup: dict[str, object] = {"network_probes": {}, "consumer": None, "nodes": {}}
    errors = []
    if campaign.network_probes:
        try:
            cleanup["network_probes"] = campaign.stop_network_probes(cleanup=True)
        except BaseException as error:
            errors.append(f"network probes: {error!r}")
    if campaign.consumer_launched:
        try:
            cleanup["consumer"] = campaign.consumer_request("exit", timeout=120, cleanup=True)
        except BaseException as error:
            cleanup["consumer"] = {"error": repr(error)}
            errors.append(f"consumer: {error!r}")
    for name in sorted(campaign.launched_nodes):
        try:
            cleanup["nodes"][name] = campaign.node_request(name, "exit", timeout=60, cleanup=True)
        except BaseException as error:
            cleanup["nodes"][name] = {"error": repr(error)}
            errors.append(f"{name}: {error!r}")
    if not errors:
        try:
            cleanup["postflight"] = campaign.postflight()
            cleanup["resource_locks"] = campaign.release_resource_locks()
        except BaseException as error:
            errors.append(f"postflight: {error!r}")
    if errors:
        cleanup["errors"] = errors
    return cleanup


def run(root: Path, approved_input_manifest_sha256: str) -> dict:
    settings = json.loads((root / "inputs/settings.json").read_text())
    fake_proxy = settings["proxy_readiness"]["fake_proxy_url"]
    for name in PROXY_ENVIRONMENT_VARIABLES:
        os.environ[name] = fake_proxy
    os.environ["NO_PROXY"] = ""
    os.environ["no_proxy"] = ""

    campaign = Campaign(root, approved_input_manifest_sha256)
    campaign.output = root / "proxy-readiness-1"
    campaign.state.update(
        state="running",
        mode="proxy_readiness",
        formal_runs_started=0,
        inherited_proxy_control={
            "proxy_variables_present": list(PROXY_ENVIRONMENT_VARIABLES),
            "no_proxy": {"NO_PROXY": "", "no_proxy": ""},
        },
    )
    campaign.output.mkdir(exist_ok=False)
    lock = (root / ".proxy-readiness.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    campaign.save()
    try:
        campaign.state["remote_inputs"] = campaign.verify_remote_inputs()
        campaign.state["resource_locks"] = campaign.acquire_resource_locks()
        campaign.phase_deadline = (
            campaign.started + campaign.settings["budgets"]["listener_preflight_seconds"]
        )
        campaign.state["network_preflight"] = campaign.network_preflight()
        campaign.state["resource_gate"] = campaign.resource_gate()
        campaign.save()

        for node in campaign.settings["nodes"]:
            campaign.launch_node(node)
        names = list(campaign.nodes)
        statuses = campaign.wait_cluster(names, timeout=60)
        listeners = campaign.listener_evidence()
        snapshot = campaign.snapshot("bootstrap")
        versions = {}
        for name, node in campaign.nodes.items():
            version = campaign.api(node["client_url"], "/version")
            if version.get("etcdserver") != campaign.settings["etcd"]["version"]:
                raise RuntimeError(f"{name} self-version differs: {version}")
            versions[name] = version

        inherited = requests.Session()
        try:
            response = inherited.get(
                next(iter(campaign.nodes.values()))["client_url"] + "/version",
                timeout=2,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            negative = {"rejected": True, "error_type": type(error).__name__}
        else:
            raise RuntimeError("inherited fake proxy unexpectedly reached etcd")
        finally:
            inherited.close()

        expected_environment = direct_network_environment_evidence(
            campaign.environment, campaign.settings["network_environment"]
        )
        child_environments = {}
        for name, node in campaign.nodes.items():
            launcher = campaign.remote_json(node, root / f"{name}.agent.launcher.json")
            agent = campaign.remote_json(node, root / name / "agent-status.json")
            child = agent.get("child", {})
            if any(
                record.get("network_environment") != expected_environment
                for record in (launcher, agent, child)
            ):
                raise RuntimeError(f"{name} direct network environment differs")
            child_environments[name] = {
                "launcher": launcher["network_environment"],
                "agent": agent["network_environment"],
                "etcd": child["network_environment"],
            }

        consumer = campaign.launch_consumer()
        manager_listener = campaign.manager_listener_evidence(
            {
                "pid": consumer["manager_pid"],
                "start_ticks": consumer["manager_start_ticks"],
            },
            peer_port=campaign.settings["orbitkv_runtime"]["manager_peer_port"],
            label="consumer Manager",
        )
        local = campaign.consumer_request("local_baseline")
        consumer_snapshot = campaign.snapshot("readiness-consumer")

        campaign.cleanup_deadline = (
            time.monotonic() + campaign.settings["budgets"]["cleanup_seconds"]
        )
        cleanup = {
            "consumer": campaign.consumer_request("exit", timeout=120, cleanup=True),
            "nodes": {},
        }
        for name in names:
            cleanup["nodes"][name] = campaign.node_request(name, "exit", timeout=60, cleanup=True)
        cleanup["postflight"] = campaign.postflight()
        cleanup["resource_locks"] = campaign.release_resource_locks()
        if any(record.get("status") != "released" for record in cleanup["resource_locks"].values()):
            raise RuntimeError(f"resource locks did not release: {cleanup}")
        result = {
            "status": "passed",
            "formal_runs_started": 0,
            "statuses": statuses,
            "listeners": listeners,
            "snapshot": snapshot,
            "versions": versions,
            "inherited_proxy_negative": negative,
            "direct_environment": expected_environment,
            "child_environments": child_environments,
            "consumer": consumer,
            "manager_listener": manager_listener,
            "local": local,
            "consumer_snapshot": consumer_snapshot,
            "cleanup": cleanup,
        }
        atomic_json(campaign.output / "result.json", result)
        campaign.state.update(state="completed", result=result, cleanup=cleanup)
        campaign.save()
        return result
    except BaseException as error:
        campaign.state.update(state="invalid", error=repr(error))
        campaign.save()
        cleanup = _cleanup(campaign)
        campaign.state["cleanup"] = cleanup
        if cleanup.get("errors"):
            campaign.state["state"] = "timeout_quarantined"
        campaign.save()
        atomic_json(campaign.output / "failure-cleanup.json", cleanup)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--approved-input-manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.approved_input_manifest_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
