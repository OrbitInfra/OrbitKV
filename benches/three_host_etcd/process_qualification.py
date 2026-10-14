"""Three-host etcd process faults, separate from Manager/cache qualification."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path

import requests

try:
    from .campaign import Campaign, atomic_json
except ImportError:
    from campaign import Campaign, atomic_json


class ProcessQualification(Campaign):
    """Consume the existing node, network and lock owners without a GPU consumer."""

    def api(
        self, endpoint: str, path: str, *, payload: dict | None = None, timeout: float = 5
    ) -> dict:
        allowance = min(timeout, self.remaining())
        if not hasattr(self, "endpoint_member_ids"):
            self.endpoint_member_ids = {}
        self.command_sequence += 1
        raw = self.output / "http"
        raw.mkdir(exist_ok=True)
        prefix = raw / f"{self.command_sequence:05d}"
        atomic_json(
            prefix.with_suffix(".request.json"),
            {"endpoint": endpoint, "path": path, "payload": payload, "timeout": allowance},
        )
        try:
            if payload is None:
                response = self.session.get(endpoint + path, timeout=allowance)
            else:
                response = self.session.post(endpoint + path, json=payload, timeout=allowance)
            prefix.with_suffix(".response.txt").write_text(response.text)
            atomic_json(
                prefix.with_suffix(".status.json"),
                {"status_code": response.status_code, "headers": dict(response.headers)},
            )
            response.raise_for_status()
            data = response.json()
            header = data.get("header")
            if header is not None:
                cluster = str(header.get("cluster_id", "0"))
                if self.cluster_id is not None and cluster != self.cluster_id:
                    raise RuntimeError("HTTP endpoint cluster identity changed")
                if (
                    self.cluster_id is not None
                    and endpoint in self.endpoint_member_ids
                    and str(header.get("member_id", "0")) != self.endpoint_member_ids[endpoint]
                ):
                    raise RuntimeError("HTTP endpoint member identity changed")
            return data
        except BaseException as error:
            atomic_json(prefix.with_suffix(".error.json"), {"error": repr(error)})
            raise

    def checkpoint(self, label: str, evidence: dict) -> None:
        self.remaining()
        locks = self.verify_resource_locks()
        expected = self.settings["process_qualification"]["phases"]
        if label != expected[len(self.state["phases"])]:
            raise RuntimeError("process phase order differs")
        self.state["phases"].append({"phase": label, "evidence": evidence, "locks": locks})
        self.save()

    def read_exact(self, endpoint: str, expected: dict[str, str]) -> dict:
        rows = self.range_prefix(
            self.settings["process_qualification"]["prefix"], endpoint=endpoint
        )
        cluster = str(rows["header"]["cluster_id"])
        revision = int(rows["header"]["revision"])
        if revision <= 0 or revision < self.last_write_revision:
            raise RuntimeError("etcd oracle revision precedes acknowledged writes")
        if cluster != self.cluster_id:
            raise RuntimeError("etcd result cluster identity differs")
        observed = {
            base64.b64decode(row["key"]).decode(): base64.b64decode(row["value"]).decode()
            for row in rows.get("kvs", [])
        }
        if observed != expected:
            raise RuntimeError(f"complete process oracle differs: {observed} != {expected}")
        return rows

    def cleanup_processes(self) -> dict:
        self.remaining(cleanup=True)
        cleanup = {}
        safe = self.state["state"] != "timeout_quarantined"
        if not safe:
            return {"safe": False, "status": "quarantined", "locks": dict(self.resource_locks)}
        if self.network_probes:
            try:
                cleanup["network_probes"] = self.stop_network_probes(cleanup=True)
                if any(
                    row.get("status") != "stopped" for row in cleanup["network_probes"].values()
                ):
                    safe = False
            except BaseException as error:
                cleanup["network_probes"] = {"status": "quarantined", "error": repr(error)}
                safe = False
        for name in sorted(self.launched_nodes):
            try:
                cleanup[name] = self.node_request(name, "exit", cleanup=True)
                self.wait_process_absent(
                    self.nodes[name], self.state["node_agents"][name], cleanup=True
                )
                self.require_ports_without_listeners(
                    self.nodes[name],
                    {self.settings["etcd"]["client_port"], self.settings["etcd"]["peer_port"]},
                    context="etcd process postflight",
                    cleanup=True,
                )
            except BaseException as error:
                cleanup[name] = {"status": "quarantined", "error": repr(error)}
                safe = False
        if safe:
            cleanup["resource_locks"] = self.release_resource_locks()
        else:
            cleanup["resource_locks"] = {
                name: {**row, "status": "retained"} for name, row in self.resource_locks.items()
            }
        cleanup["safe"] = safe and all(
            row.get("status") == "released" for row in cleanup["resource_locks"].values()
        )
        return cleanup

    def run_process(self, *, readiness: bool = False) -> dict:
        if not readiness:
            self.validate_launch_authorization()
        if self.settings["process_qualification"].get("phases") != [
            "bootstrap",
            "leader_loss_rejoin",
            "quorum_loss_recovery",
            "complete_final_puts_deletes",
        ]:
            raise RuntimeError("process cell list differs")
        if self.settings.get("network_environment") != {
            "unset": [
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "ALL_PROXY",
                "http_proxy",
                "https_proxy",
                "all_proxy",
            ],
            "no_proxy": "*",
        }:
            raise RuntimeError("process network environment contract differs")
        if (
            self.settings["etcd"]["sha256"]
            != hashlib.sha256((self.root / "frozen/etcd").read_bytes()).hexdigest()
        ):
            raise RuntimeError("process etcd binary differs from frozen digest")
        contract = self.settings["process_qualification"]
        if (
            set(contract) != {"scope", "formal_launches", "phases", "keys", "prefix"}
            or type(contract["keys"]) is not int
            or contract["keys"] != 32
            or not isinstance(contract["prefix"], str)
            or not contract["prefix"].startswith("/orbitkv-s2-process-")
            or not contract["prefix"].endswith("/")
        ):
            raise RuntimeError("invalid frozen process oracle")
        if contract["scope"] != "three_host_etcd_process_only" or contract["formal_launches"] != 1:
            raise RuntimeError("invalid process-only qualification contract")
        existing = self.root / ("process-readiness-1" if readiness else "process-qualification-1")
        if existing.exists() or (self.root / ".process-qualification.lock").exists():
            raise RuntimeError("process qualification already launched; no retry")
        self.output = existing
        self.output.mkdir(exist_ok=False)
        lock = (self.root / ".process-qualification.lock").open("x")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.endpoint_member_ids = {}
        self.state.update(
            state="running",
            formal_runs_started=0 if readiness else 1,
            scope=contract["scope"],
            readiness=readiness,
        )
        self.save()
        expected = {}
        names = list(self.nodes)
        try:
            self.verify_remote_inputs()
            self.acquire_resource_locks()
            self.state["resource_lock_checks"] = self.verify_resource_locks()
            for node in self.nodes.values():
                self.require_ports_without_listeners(
                    node,
                    {self.settings["etcd"]["client_port"], self.settings["etcd"]["peer_port"]},
                    context="process-only resource preflight",
                )
            network = self.network_preflight()
            for node in self.settings["nodes"]:
                self.launch_node(node)
            statuses = self.wait_cluster(names, timeout=60)
            clusters = {str(row["header"]["cluster_id"]) for row in statuses.values()}
            member_ids = {str(row["header"]["member_id"]) for row in statuses.values()}
            if len(clusters) != 1 or "0" in clusters or len(member_ids) != 3 or "0" in member_ids:
                raise RuntimeError("executing cluster/member identities differ")
            self.cluster_id = next(iter(clusters))
            self.endpoint_member_ids = {
                self.nodes[name]["client_url"]: str(row["header"]["member_id"])
                for name, row in statuses.items()
            }
            members = self.api(self.endpoint(), "/v3/cluster/member/list", payload={})
            expected_members = {
                (node["name"], node["peer_url"], node["client_url"]) for node in self.nodes.values()
            }
            actual_members = {
                (row["name"], row["peerURLs"][0], row["clientURLs"][0])
                for row in members.get("members", [])
                if len(row.get("peerURLs", [])) == len(row.get("clientURLs", [])) == 1
            }
            member_ids_by_name = {row["name"]: str(row["ID"]) for row in members.get("members", [])}
            if any(
                member_ids_by_name.get(name) != self.endpoint_member_ids[node["client_url"]]
                for name, node in self.nodes.items()
            ):
                raise RuntimeError("endpoint and member-list identity differ")
            if len(members.get("members", [])) != 3 or actual_members != expected_members:
                raise RuntimeError("process cluster member/endpoint identity differs")
            versions = {
                name: self.api(node["client_url"], "/version") for name, node in self.nodes.items()
            }
            if any(
                row.get("etcdserver") != self.settings["etcd"]["version"]
                for row in versions.values()
            ):
                raise RuntimeError("executing etcd version differs")
            self.checkpoint(
                "bootstrap",
                {
                    "statuses": statuses,
                    "members": members,
                    "versions": versions,
                    "network": network,
                    "listeners": self.listener_evidence(),
                },
            )
            if readiness:
                probes = {}
                for source in self.nodes.values():
                    for target in self.nodes.values():
                        if source["name"] == target["name"]:
                            continue
                        command = [
                            "python3",
                            "-c",
                            "import json,requests,sys; s=requests.Session(); s.trust_env=False; r=s.get(sys.argv[1]+'/version', timeout=3); r.raise_for_status(); print(json.dumps(r.json()))",
                            target["client_url"],
                        ]
                        result = self.host_command(source, command, timeout=10)
                        value = json.loads(result.stdout)
                        if value.get("etcdserver") != self.settings["etcd"]["version"]:
                            raise RuntimeError("six-direction executing client version differs")
                        probes[source["name"] + "->" + target["name"]] = value
                cleanup = self.cleanup_processes()
                if not cleanup.get("safe"):
                    raise RuntimeError("readiness cleanup did not finish")
                result = {
                    "state": "readiness_completed",
                    "formal_runs_started": 0,
                    "scope": contract["scope"],
                    "bootstrap": self.state["phases"][0],
                    "direct_http": probes,
                    "cleanup": cleanup,
                }
                atomic_json(self.output / "result.json", result)
                self.state.update(state="readiness_completed", result=result)
                self.save()
                return result
            prefix = contract["prefix"]
            for index in range(contract["keys"]):
                key = f"{prefix}{index:04d}"
                value = f"initial-{index:04d}"
                self.put(key, value)
                expected[key] = value
            before = self.wait_cluster(names)
            leader = self.leader(before)
            old_term = int(before[leader]["raftTerm"])
            stopped = self.node_request(leader, "stop")
            survivors = [name for name in names if name != leader]
            after = self.wait_cluster(survivors)
            new_leader = self.leader(after)
            if new_leader == leader or int(after[new_leader]["raftTerm"]) <= old_term:
                raise RuntimeError("leader loss did not elect a new higher-term leader")
            marker = prefix + "leader-recovered"
            self.put(marker, "committed", endpoint=self.endpoint({leader}))
            expected[marker] = "committed"
            self.node_request(leader, "start")
            self.wait_cluster(names)
            leader_oracle = {
                name: self.read_exact(node["client_url"], expected)
                for name, node in self.nodes.items()
            }
            self.checkpoint(
                "leader_loss_rejoin",
                {
                    "old_leader": leader,
                    "new_leader": new_leader,
                    "stop": stopped,
                    "oracle": leader_oracle,
                    "listeners": self.listener_evidence(),
                },
            )
            quorum_leader = self.leader(self.wait_cluster(names))
            stopped_names = [quorum_leader, next(name for name in names if name != quorum_leader)]
            stops = {name: self.node_request(name, "stop") for name in stopped_names}
            remaining = next(name for name in names if name not in stopped_names)
            try:
                self.put(
                    prefix + "must-not-commit",
                    "invalid",
                    endpoint=self.nodes[remaining]["client_url"],
                    timeout=3,
                )
            except requests.RequestException as error:
                rejection = repr(error)
            else:
                raise RuntimeError("write committed without quorum")
            for name in stopped_names:
                self.node_request(name, "start")
            self.wait_cluster(names)
            quorum_oracle = {
                name: self.read_exact(node["client_url"], expected)
                for name, node in self.nodes.items()
            }
            self.checkpoint(
                "quorum_loss_recovery",
                {
                    "stops": stops,
                    "write_rejected": rejection,
                    "oracle": quorum_oracle,
                    "listeners": self.listener_evidence(),
                },
            )
            endpoint = self.endpoint()
            for index in range(contract["keys"]):
                key = f"{prefix}{index:04d}"
                if index % 2:
                    self.api(
                        endpoint,
                        "/v3/kv/deleterange",
                        payload={"key": base64.b64encode(key.encode()).decode()},
                    )
                    expected.pop(key)
                else:
                    expected[key] = f"final-{index:04d}"
                    self.put(key, expected[key], endpoint=endpoint)
            final = {
                name: self.read_exact(node["client_url"], expected)
                for name, node in self.nodes.items()
            }
            self.checkpoint("complete_final_puts_deletes", {"oracle": final, "expected": expected})
            cleanup = self.cleanup_processes()
            if not cleanup["safe"]:
                raise RuntimeError("process cleanup is incomplete or quarantined")
            lock_cleanup = cleanup["resource_locks"]
            result = {
                "state": "completed_pending_review",
                "scope": contract["scope"],
                "formal_runs_started": 1,
                "expected": expected,
                "final": final,
                "cleanup": cleanup,
                "resource_locks": lock_cleanup,
                "no_gpu_or_cache_started": True,
            }
            atomic_json(self.output / "result.json", result)
            self.state.update(state="completed_pending_review", result=result)
            self.save()
            return result
        except BaseException as error:
            timed_out = self.state["state"] == "timeout_quarantined"
            self.state.update(
                state="timeout_quarantined" if timed_out else "invalid_no_retry", error=repr(error)
            )
            try:
                cleanup = self.cleanup_processes()
            except BaseException as cleanup_error:
                cleanup = {
                    "safe": False,
                    "error": repr(cleanup_error),
                    "locks": dict(self.resource_locks),
                }
            self.state["cleanup"] = cleanup
            if not cleanup.get("safe"):
                self.state["state"] = "timeout_quarantined"
            self.save()
            atomic_json(self.output / "failure-cleanup.json", cleanup)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--approved-input-manifest-sha256", required=True)
    parser.add_argument("--launch-authorization", type=Path)
    parser.add_argument("--approved-launch-authorization-sha256")
    parser.add_argument("--readiness-only", action="store_true")
    args = parser.parse_args()
    campaign = ProcessQualification(
        args.root,
        args.approved_input_manifest_sha256,
        args.launch_authorization,
        args.approved_launch_authorization_sha256,
    )
    print(json.dumps(campaign.run_process(readiness=args.readiness_only), sort_keys=True))


if __name__ == "__main__":
    main()
