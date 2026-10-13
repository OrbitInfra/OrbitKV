"""Persistent fail-stop three-host etcd process/leader/quorum campaign."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import ipaddress
import json
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path

import requests

try:
    from .environment import direct_network_environment, direct_network_environment_evidence
    from .integrity import sha256, verify_inputs
except ImportError:
    from environment import direct_network_environment, direct_network_environment_evidence
    from integrity import sha256, verify_inputs


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prefix_end(value: bytes) -> bytes:
    result = bytearray(value)
    for index in range(len(result) - 1, -1, -1):
        if result[index] != 0xFF:
            result[index] += 1
            return bytes(result[: index + 1])
    return b"\0"


def start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class Campaign:
    def __init__(
        self,
        root: Path,
        approved_input_manifest_sha256: str,
        launch_authorization_path: Path | None = None,
        approved_launch_authorization_sha256: str | None = None,
    ):
        self.root = root
        self.approved_input_manifest_sha256 = approved_input_manifest_sha256
        self.launch_authorization_path = launch_authorization_path
        self.approved_launch_authorization_sha256 = approved_launch_authorization_sha256
        self.verified_inputs = verify_inputs(root, approved_input_manifest_sha256)
        self.settings = json.loads((root / "inputs/settings.json").read_text())
        self.environment = direct_network_environment(self.settings["network_environment"])
        self.output = root / "campaign-1"
        self.started = time.monotonic()
        self.session = requests.Session()
        self.session.trust_env = False
        self.sequence = 0
        self.command_sequence = 0
        self.listener_observation = 0
        self.phase_deadline: float | None = None
        self.cleanup_deadline: float | None = None
        self.nodes = {node["name"]: node for node in self.settings["nodes"]}
        self.launched_nodes = set()
        self.consumer_launched = False
        self.resource_locks: dict[str, dict] = {}
        self.network_probes: dict[str, dict] = {}
        self.last_revision = 0
        self.last_write_revision = 0
        self.cluster_id: str | None = None
        self.initial_member_ids: dict[str, str] | None = None
        self.format_uuid: uuid.UUID | None = None
        self.node_epochs: dict[str, int] = {}
        self.node_incarnations: dict[str, str] = {}
        self.retired_incarnations: dict[str, set[str]] = {}
        self.prefault_member_evidence: dict[str, dict] = {}
        runtime = self.settings["orbitkv_runtime"]
        self.expected_orbitkv_endpoints = {
            "a100-consumer": f"127.0.0.1:{runtime['manager_peer_port']}",
            "query-source": f"127.0.0.1:{runtime['query_source_peer_port']}",
            "cold-observer": f"127.0.0.1:{runtime['observer_peer_port']}",
        }
        self.required_orbitkv_members = {
            "a100-consumer",
            "metadata-pressure",
            "query-source",
        }
        self.local_payload_sha256: str | None = None
        self.owner_watermarks: dict[str, int] = {}
        self.state = {
            "state": "preflight",
            "controller_pid": __import__("os").getpid(),
            "controller_start_ticks": start_ticks(__import__("os").getpid()),
            "started_unix_ns": time.time_ns(),
            "phases": [],
            "current": None,
            "node_agents": {},
            "consumer": None,
            "approved_input_manifest_sha256": approved_input_manifest_sha256,
            "input_files": sorted(self.verified_inputs),
            "resource_locks": self.resource_locks,
            "network_probes": self.network_probes,
            "quarantined": [],
            "active_command": None,
            "network_environment": direct_network_environment_evidence(
                self.environment, self.settings["network_environment"]
            ),
        }

    def validate_launch_authorization(self) -> None:
        path = self.resolve_stage_file(
            self.launch_authorization_path, "formal launch authorization"
        )
        approved_sha256 = self.approved_launch_authorization_sha256
        if (
            not isinstance(approved_sha256, str)
            or len(approved_sha256) != 64
            or sha256(path) != approved_sha256
        ):
            raise PermissionError("formal launch authorization identity differs")
        authorization = json.loads(path.read_text())
        expected = {
            "formal_launch_authorized": True,
            "formal_runs_started": 0,
            "task_id": self.settings["task_id"],
            "source_commit": self.settings["source_commit"],
            "source_tree": self.settings["source_tree"],
            "input_manifest_sha256": self.approved_input_manifest_sha256,
            "launch_review_verdict": "ACCEPTED",
        }
        if not isinstance(authorization, dict) or any(
            authorization.get(key) != value for key, value in expected.items()
        ):
            raise PermissionError(
                f"formal launch lacks exact independent authorization: {authorization}"
            )
        review = self.resolve_stage_file(
            Path(authorization.get("launch_review_path", "")),
            "independent launch review",
        )
        review_sha256 = authorization.get("launch_review_sha256")
        verdict_path = self.resolve_stage_file(
            Path(authorization.get("launch_review_verdict_path", "")),
            "independent launch verdict",
        )
        verdict_sha256 = authorization.get("launch_review_verdict_sha256")
        if (
            not isinstance(review_sha256, str)
            or len(review_sha256) != 64
            or sha256(review) != review_sha256
            or not isinstance(verdict_sha256, str)
            or len(verdict_sha256) != 64
            or sha256(verdict_path) != verdict_sha256
        ):
            raise PermissionError("independent launch review identity differs")
        first_line = next(
            (line.strip() for line in review.read_text().splitlines() if line.strip()),
            "",
        )
        verdict = json.loads(verdict_path.read_text())
        if (
            not first_line.startswith("ACCEPTED")
            or verdict.get("verdict") != "ACCEPTED"
            or verdict.get("accepted") is not True
            or verdict.get("candidate_commit") != self.settings["source_commit"]
            or verdict.get("candidate_tree") != self.settings["source_tree"]
            or verdict.get("input_manifest_sha256") != self.approved_input_manifest_sha256
        ):
            raise PermissionError("independent launch review is not accepted")

    def resolve_stage_file(self, path: Path | None, label: str) -> Path:
        if not isinstance(path, Path) or not path.is_absolute():
            raise PermissionError(f"{label} path is not absolute")
        try:
            root = self.root.resolve(strict=True)
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise PermissionError(f"{label} path is unavailable: {path}") from error
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise PermissionError(f"{label} path escapes the frozen stage: {path}")
        return resolved

    def complete_success(self, final_evidence: dict, cleanup: dict, postflight: dict) -> None:
        final_evidence["cleanup"] = dict(cleanup)
        final_evidence["postflight"] = postflight
        self.complete_phase("cold_observer_final_oracle", final_evidence)

        resource_lock_cleanup = self.release_resource_locks()
        expected_locks = set(self.resource_locks)
        if set(resource_lock_cleanup) != expected_locks or any(
            record.get("status") != "released"
            or record.get("pid") != self.resource_locks[name].get("pid")
            or record.get("start_ticks") != self.resource_locks[name].get("start_ticks")
            for name, record in resource_lock_cleanup.items()
        ):
            raise RuntimeError(f"resource locks did not release: {resource_lock_cleanup}")
        atomic_json(self.output / "resource-lock-release.json", resource_lock_cleanup)
        cleanup["resource_locks"] = resource_lock_cleanup
        atomic_json(self.output / "cleanup.json", cleanup)
        self.state["state"] = "completed_pending_review"
        self.state["current"] = None
        self.state["cleanup"] = cleanup
        self.state["postflight"] = postflight
        self.state["resource_lock_release"] = resource_lock_cleanup
        self.save()

    def remaining(self, *, cleanup: bool = False) -> float:
        if cleanup:
            if self.cleanup_deadline is None:
                self.cleanup_deadline = (
                    time.monotonic() + self.settings["budgets"]["cleanup_seconds"]
                )
                self.state["cleanup_deadline_mono"] = self.cleanup_deadline
                self.save()
            value = self.cleanup_deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("three-host etcd cleanup watchdog exhausted")
            return value
        global_remaining = self.settings["budgets"]["global_seconds"] - (
            time.monotonic() - self.started
        )
        phase_remaining = (
            self.phase_deadline - time.monotonic()
            if self.phase_deadline is not None
            else global_remaining
        )
        value = min(global_remaining, phase_remaining)
        if value <= 0:
            raise TimeoutError("three-host etcd global watchdog exhausted")
        return value

    def save(self) -> None:
        self.state["updated_unix_ns"] = time.time_ns()
        atomic_json(self.output / "status.json", self.state)

    def log_event(self, event: dict) -> None:
        event = {"observed_unix_ns": time.time_ns(), **event}
        with (self.output / "timeline.jsonl").open("a") as log:
            log.write(json.dumps(event, sort_keys=True) + "\n")

    def host_command(
        self,
        node: dict,
        command: list[str],
        *,
        timeout: float | None = None,
        check: bool = True,
        cleanup: bool = False,
    ) -> subprocess.CompletedProcess:
        if node["ssh"]:
            argv = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                node["ssh"],
                shlex.join(command),
            ]
        else:
            argv = command
        allowance = min(
            timeout or self.settings["budgets"]["command_seconds"],
            self.remaining(cleanup=cleanup),
        )
        started = time.time_ns()
        self.command_sequence += 1
        command_root = self.output / "command-output"
        command_root.mkdir(exist_ok=True)
        stdout_path = command_root / f"{self.command_sequence:05d}.stdout"
        stderr_path = command_root / f"{self.command_sequence:05d}.stderr"
        stdout = stdout_path.open("x")
        stderr = stderr_path.open("x")
        process = subprocess.Popen(
            argv,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        stdout.close()
        stderr.close()
        active = {
            "node": node["name"],
            "pid": process.pid,
            "start_ticks": start_ticks(process.pid),
            "process_group": process.pid,
            "command": argv,
            "started_unix_ns": started,
            "cleanup": cleanup,
        }
        self.state["active_command"] = active
        self.save()
        try:
            returncode = process.wait(timeout=allowance)
        except subprocess.TimeoutExpired:
            record = {
                **active,
                "status": "quarantined",
                "timeout_seconds": allowance,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "observed_unix_ns": time.time_ns(),
            }
            self.state["quarantined"].append(record)
            self.state["state"] = "timeout_quarantined"
            self.save()
            raise TimeoutError(f"host command quarantined: {record}") from None
        stdout_text = stdout_path.read_text(errors="replace")
        stderr_text = stderr_path.read_text(errors="replace")
        record = {
            "node": node["name"],
            "command": argv,
            "timeout_seconds": allowance,
            "started_unix_ns": started,
            "finished_unix_ns": time.time_ns(),
            "exit_code": returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        with (self.output / "commands.jsonl").open("a") as log:
            log.write(json.dumps(record, sort_keys=True) + "\n")
        self.state["active_command"] = None
        self.save()
        if check and returncode != 0:
            raise RuntimeError(f"host command failed: {record}")
        return subprocess.CompletedProcess(argv, returncode, stdout_text, stderr_text)

    def remote_json(self, node: dict, path: Path, *, cleanup: bool = False) -> dict:
        result = self.host_command(node, ["cat", str(path)], cleanup=cleanup)
        return json.loads(result.stdout)

    def wait_remote_json(
        self, node: dict, path: Path, timeout: float = 30, *, cleanup: bool = False
    ) -> dict:
        deadline = time.monotonic() + min(timeout, self.remaining(cleanup=cleanup))
        while time.monotonic() < deadline:
            result = self.host_command(
                node,
                ["test", "-f", str(path)],
                timeout=10,
                check=False,
                cleanup=cleanup,
            )
            if result.returncode == 0:
                return self.remote_json(node, path, cleanup=cleanup)
            time.sleep(0.2)
        raise TimeoutError(f"timed out waiting for {node['name']}:{path}")

    def touch(self, node: dict, path: Path, *, cleanup: bool = False) -> None:
        self.host_command(
            node,
            [
                "python3",
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).touch(exist_ok=False)",
                str(path),
            ],
            cleanup=cleanup,
        )

    def acquire_resource_locks(self) -> dict[str, dict]:
        for node in self.settings["nodes"]:
            name = node["name"]
            self.resource_locks[name] = {
                "status": "launching",
                "ownership": "remote lock agent may exist after command dispatch",
            }
            self.save()
            result = self.host_command(
                node,
                [
                    "python3",
                    str(self.root / "inputs/launch_agent.py"),
                    "--root",
                    str(self.root),
                    "--kind",
                    "resource-lock",
                    "--node",
                    name,
                    "--approved-input-manifest-sha256",
                    self.approved_input_manifest_sha256,
                ],
            )
            launch = json.loads(result.stdout)
            ready = self.wait_remote_json(
                node, self.root / f"{name}.resource-lock.json", timeout=30
            )
            if (
                ready.get("status") != "locked"
                or ready.get("task_id") != self.settings["task_id"]
                or ready.get("node") != name
                or ready.get("hostname") != node["hostname"]
                or ready.get("boot_id") != node["boot_id"]
                or ready.get("pid") != launch.get("pid")
                or ready.get("start_ticks") != launch.get("start_ticks")
                or not isinstance(ready.get("start_ticks"), str)
                or not ready["start_ticks"]
                or ready.get("approved_input_manifest_sha256")
                != self.approved_input_manifest_sha256
            ):
                raise RuntimeError(f"resource lock identity differs: {launch}, {ready}")
            self.resource_locks[name] = ready
            self.save()
        return dict(self.resource_locks)

    def verify_remote_inputs(self) -> dict[str, dict]:
        evidence = {}
        for node in self.settings["nodes"]:
            digest = self.host_command(
                node, ["sha256sum", str(self.root / "inputs/SHA256SUMS")]
            ).stdout.split()[0]
            if digest != self.approved_input_manifest_sha256:
                raise RuntimeError(f"{node['name']} input manifest differs: {digest}")
            self.host_command(
                node,
                [
                    "python3",
                    "-c",
                    (
                        "import os,subprocess,sys; os.chdir(sys.argv[1]); "
                        "raise SystemExit(subprocess.run(['sha256sum','--check','--strict','--quiet','SHA256SUMS']).returncode)"
                    ),
                    str(self.root / "inputs"),
                ],
            )
            evidence[node["name"]] = {
                "manifest_sha256": digest,
                "files": sorted(self.verified_inputs),
            }
        return evidence

    def release_resource_locks(self) -> dict[str, dict]:
        cleanup: dict[str, dict] = {}
        for name, record in list(self.resource_locks.items()):
            node = self.nodes[name]
            try:
                if record.get("status") == "launching":
                    ready_path = self.root / f"{name}.resource-lock.json"
                    probe = self.host_command(
                        node,
                        ["test", "-f", str(ready_path)],
                        check=False,
                        cleanup=True,
                    )
                    if probe.returncode != 0:
                        cleanup[name] = {
                            "status": "unknown_launch",
                            "quarantined": True,
                        }
                        continue
                    record = self.remote_json(node, ready_path, cleanup=True)
                self.touch(
                    node,
                    self.root / f"{name}.resource-lock.stop",
                    cleanup=True,
                )
                released = self.wait_remote_json(
                    node,
                    self.root / f"{name}.resource-lock.released.json",
                    timeout=30,
                    cleanup=True,
                )
                if (
                    released.get("status") != "released"
                    or released.get("pid") != record.get("pid")
                    or released.get("start_ticks") != record.get("start_ticks")
                ):
                    raise RuntimeError(f"resource lock release differs: {released}")
                self.wait_process_absent(node, record)
                cleanup[name] = released
            except BaseException as error:
                cleanup[name] = {
                    "status": "error",
                    "error": repr(error),
                    "pid": record.get("pid"),
                    "start_ticks": record.get("start_ticks"),
                    "quarantined": True,
                }
        self.state["resource_lock_cleanup"] = cleanup
        self.save()
        return cleanup

    def verify_resource_locks(self) -> dict[str, dict]:
        evidence = {}
        for name, record in self.resource_locks.items():
            if record.get("status") != "locked":
                raise RuntimeError(f"resource lock is not held: {name}: {record}")
            node = self.nodes[name]
            current = self.remote_json(node, self.root / f"{name}.resource-lock.json")
            identity = self.host_command(
                node,
                [
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path; import sys; p=Path('/proc/'+sys.argv[1]+'/stat'); "
                        "sys.exit(0 if p.exists() and p.read_text().rsplit(')', 1)[1].split()[19] == sys.argv[2] else 1)"
                    ),
                    str(record["pid"]),
                    record["start_ticks"],
                ],
                check=False,
            )
            lock_file = self.host_command(
                node,
                [
                    "python3",
                    "-c",
                    "import os,sys; s=os.lstat(sys.argv[1]); sys.exit(0 if (s.st_dev,s.st_ino,s.st_uid)==tuple(map(int,sys.argv[2:])) else 1)",
                    record["lock_path"],
                    str(record["lock_device"]),
                    str(record["lock_inode"]),
                    str(record["lock_uid"]),
                ],
                check=False,
            )
            if identity.returncode != 0 or lock_file.returncode != 0 or current != record:
                raise RuntimeError(
                    f"resource lock ownership changed: {name}: {current} != {record}"
                )
            evidence[name] = current
        return evidence

    def network_preflight(self) -> dict:
        for node in self.settings["nodes"]:
            name = node["name"]
            self.network_probes[name] = {"status": "launching"}
            self.save()
            launched = self.host_command(
                node,
                [
                    "python3",
                    str(self.root / "inputs/launch_agent.py"),
                    "--root",
                    str(self.root),
                    "--kind",
                    "network-probe",
                    "--node",
                    name,
                    "--approved-input-manifest-sha256",
                    self.approved_input_manifest_sha256,
                ],
            )
            launch = json.loads(launched.stdout)
            ready = self.wait_remote_json(
                node, self.root / f"{name}.network-probe.json", timeout=30
            )
            if (
                ready.get("status") != "listening"
                or ready.get("pid") != launch.get("pid")
                or ready.get("start_ticks") != launch.get("start_ticks")
                or not isinstance(ready.get("start_ticks"), str)
                or ready.get("hostname") != node["hostname"]
                or ready.get("boot_id") != node["boot_id"]
                or ready.get("ipv6") != node["ipv6"]
            ):
                raise RuntimeError(f"network probe identity differs: {launch}, {ready}")
            self.network_probes[name] = ready
            self.save()

        directed = []
        for source in self.settings["nodes"]:
            for destination in self.settings["nodes"]:
                if source["name"] == destination["name"]:
                    continue
                for port in (
                    self.settings["etcd"]["client_port"],
                    self.settings["etcd"]["peer_port"],
                ):
                    self.host_command(
                        source,
                        [
                            "python3",
                            "-c",
                            (
                                "import socket,sys; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=3); "
                                "s.sendall(bytes([165])); s.close()"
                            ),
                            destination["ipv6"],
                            str(port),
                        ],
                        timeout=5,
                    )
                    directed.append(
                        {
                            "source": source["name"],
                            "source_ipv6": source["ipv6"],
                            "destination": destination["name"],
                            "destination_ipv6": destination["ipv6"],
                            "port": port,
                        }
                    )
        for node in self.settings["nodes"]:
            deadline = time.monotonic() + min(30, self.remaining())
            while True:
                observed = self.remote_json(node, self.root / f"{node['name']}.network-probe.json")
                if len(observed["connections"]) == 4:
                    break
                if len(observed["connections"]) > 4 or time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"network probe connection count differs: {node['name']}: "
                        f"{len(observed['connections'])}"
                    )
                time.sleep(0.1)
        results = self.stop_network_probes(cleanup=False)
        for destination in self.settings["nodes"]:
            expected = {
                (source["ipv6"], port)
                for source in self.settings["nodes"]
                if source["name"] != destination["name"]
                for port in (
                    self.settings["etcd"]["client_port"],
                    self.settings["etcd"]["peer_port"],
                )
            }
            observed = {
                (row["source_ipv6"], row["port"])
                for row in results[destination["name"]]["connections"]
                if row["payload_hex"] == "a5"
            }
            if observed != expected:
                raise RuntimeError(
                    f"directed network evidence differs for {destination['name']}: "
                    f"{observed} != {expected}"
                )
        return {"directed": directed, "listeners": results}

    def stop_network_probes(self, *, cleanup: bool) -> dict[str, dict]:
        results = {}
        for name, record in list(self.network_probes.items()):
            if record.get("status") == "stopped":
                results[name] = record
                continue
            node = self.nodes[name]
            if record.get("status") == "launching":
                ready_path = self.root / f"{name}.network-probe.json"
                probe = self.host_command(
                    node,
                    ["test", "-f", str(ready_path)],
                    check=False,
                    cleanup=cleanup,
                )
                if probe.returncode != 0:
                    results[name] = {
                        "status": "unknown_launch",
                        "quarantined": True,
                    }
                    continue
                record = self.remote_json(node, ready_path, cleanup=cleanup)
            self.touch(
                node,
                self.root / f"{name}.network-probe.stop",
                cleanup=cleanup,
            )
            result = self.wait_remote_json(
                node,
                self.root / f"{name}.network-probe.result.json",
                timeout=30,
                cleanup=cleanup,
            )
            if result.get("status") != "stopped":
                raise RuntimeError(f"network probe did not stop: {result}")
            self.wait_process_absent(node, record, cleanup=cleanup)
            self.network_probes[name] = result
            results[name] = result
            self.save()
        return results

    def launch_node(self, node: dict) -> dict:
        self.launched_nodes.add(node["name"])
        self.state["node_agents"][node["name"]] = {
            "status": "launching",
            "ownership": "remote launch may exist after command dispatch",
        }
        self.save()
        result = self.host_command(
            node,
            [
                "python3",
                str(self.root / "inputs/launch_agent.py"),
                "--root",
                str(self.root),
                "--kind",
                "node",
                "--node",
                node["name"],
                "--approved-input-manifest-sha256",
                self.approved_input_manifest_sha256,
            ],
        )
        launch = json.loads(result.stdout)
        status = self.wait_remote_json(
            node, self.root / node["name"] / "agent-status.json", timeout=30
        )
        if status["state"] != "running" or status["child"] is None:
            raise RuntimeError(f"node agent did not start: {status}")
        self.state["node_agents"][node["name"]] = launch
        self.save()
        return status

    def node_request(
        self, name: str, action: str, timeout: float = 60, *, cleanup: bool = False
    ) -> dict:
        node = self.nodes[name]
        self.sequence += 1
        stem = f"{self.sequence:04d}-{action}"
        remaining = self.remaining(cleanup=cleanup) if cleanup else None
        self.host_command(
            node,
            [
                "python3",
                "-c",
                "import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({'cleanup_remaining_seconds':json.loads(sys.argv[2])}))",
                str(self.root / name / "controls" / f"{stem}.request"),
                json.dumps(remaining),
            ],
            cleanup=cleanup,
        )
        result = self.wait_remote_json(
            node,
            self.root / name / "responses" / f"{stem}.json",
            timeout=timeout,
            cleanup=cleanup,
        )
        if result["status"] == "quarantined":
            self.state["quarantined"].append({"node": name, **result})
            self.save()
            raise TimeoutError(f"{name} quarantined: {result}")
        expected = {
            "stop": {"stopped", "terminated_by_sigterm"},
            "start": {"started"},
            "exit": {"stopped", "terminated_by_sigterm", "already_stopped"},
            "status": {"observed"},
        }[action]
        if result["status"] not in expected:
            raise RuntimeError(f"{name} {action} failed: {result}")
        if result["status"] == "terminated_by_sigterm":
            if result.get("returncode") != -15 or not result.get("sigterm_sent"):
                raise RuntimeError(f"{name} reported invalid SIGTERM exit: {result}")
        elif result.get("returncode") not in (None, 0):
            raise RuntimeError(f"{name} {action} returned nonzero: {result}")
        return result

    def launch_consumer(self) -> dict:
        node = self.nodes["s2-a100"]
        self.consumer_launched = True
        self.state["consumer"] = {
            "status": "launching",
            "ownership": "remote launch may exist after command dispatch",
        }
        self.save()
        result = self.host_command(
            node,
            [
                "python3",
                str(self.root / "inputs/launch_agent.py"),
                "--root",
                str(self.root),
                "--kind",
                "consumer",
                "--approved-input-manifest-sha256",
                self.approved_input_manifest_sha256,
            ],
        )
        launch = json.loads(result.stdout)
        self.state["consumer"] = launch
        self.save()
        status_path = self.root / "consumer/agent-status.json"
        deadline = time.monotonic() + min(120, self.remaining())
        status = self.wait_remote_json(node, status_path, timeout=120)
        while status.get("state") == "starting":
            self.remaining()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"consumer remained in starting state: {status}")
            time.sleep(0.1)
            status = self.remote_json(node, status_path)
        if status.get("state") != "running":
            failure_cleanup = status.get("failure_cleanup")
            cleanup_safe = (
                isinstance(failure_cleanup, dict)
                and failure_cleanup.get("quarantined") is False
                and failure_cleanup.get("uds_root_cleanup", {}).get("status")
                in {"released", "not_created"}
            )
            if cleanup_safe:
                self.wait_process_absent(node, launch, cleanup=False)
                self.consumer_launched = False
                self.state["consumer"] = {
                    **launch,
                    "status": "startup_failed_reaped",
                    "terminal": status,
                    "process_absent": True,
                }
                self.save()
            raise RuntimeError(f"consumer did not start: {status}")
        return status

    def consumer_request(self, action: str, timeout: float = 60, *, cleanup: bool = False) -> dict:
        node = self.nodes["s2-a100"]
        self.sequence += 1
        stem = f"{self.sequence:04d}-{action}"
        remaining = self.remaining(cleanup=cleanup) if cleanup else None
        self.host_command(
            node,
            [
                "python3",
                "-c",
                "import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({'cleanup_remaining_seconds':json.loads(sys.argv[2])}))",
                str(self.root / "consumer/controls" / f"{stem}.request"),
                json.dumps(remaining),
            ],
            cleanup=cleanup,
        )
        response = self.wait_remote_json(
            node,
            self.root / "consumer/responses" / f"{stem}.json",
            timeout=timeout,
            cleanup=cleanup,
        )
        if "error" in response:
            raise RuntimeError(f"consumer action {action} failed: {response}")
        return response["result"]

    def api(
        self,
        endpoint: str,
        path: str,
        *,
        payload: dict | None = None,
        timeout: float = 5,
    ) -> dict:
        url = f"{endpoint}{path}"
        allowance = min(timeout, self.remaining())
        if payload is None:
            response = self.session.get(url, timeout=allowance)
        else:
            response = self.session.post(url, json=payload, timeout=allowance)
        response.raise_for_status()
        return response.json()

    def status(self, node: dict) -> dict:
        return self.api(node["client_url"], "/v3/maintenance/status", payload={})

    def statuses(self, names: list[str]) -> dict:
        return {name: self.status(self.nodes[name]) for name in names}

    def wait_cluster(self, names: list[str], timeout: float = 30) -> dict:
        deadline = time.monotonic() + min(timeout, self.remaining())
        last = {}
        while time.monotonic() < deadline:
            try:
                last = self.statuses(names)
                leaders = {row["leader"] for row in last.values()}
                if len(leaders) == 1 and next(iter(leaders)) not in ("0", 0, None):
                    return last
            except (requests.RequestException, KeyError, ValueError) as error:
                last = {"error": repr(error)}
            time.sleep(0.2)
        raise TimeoutError(f"cluster did not become healthy: {last}")

    def leader(self, statuses: dict) -> str:
        leader = next(iter(statuses.values()))["leader"]
        for name, row in statuses.items():
            if row["header"]["member_id"] == leader:
                return name
        raise RuntimeError(f"leader {leader} not mapped: {statuses}")

    def endpoint(self, excluded: set[str] | None = None) -> str:
        excluded = excluded or set()
        for name, node in self.nodes.items():
            if name in excluded:
                continue
            try:
                self.status(node)
                return node["client_url"]
            except requests.RequestException:
                continue
        raise RuntimeError("no reachable etcd endpoint")

    def put(self, key: str, value: str, *, endpoint: str | None = None, timeout: float = 5) -> dict:
        result = self.api(
            endpoint or self.endpoint(),
            "/v3/kv/put",
            payload={
                "key": base64.b64encode(key.encode()).decode(),
                "value": base64.b64encode(value.encode()).decode(),
            },
            timeout=timeout,
        )
        revision = int(result["header"]["revision"])
        if revision <= self.last_write_revision:
            raise RuntimeError(
                f"committed revision did not advance: {revision} <= {self.last_write_revision}"
            )
        self.last_write_revision = revision
        return result

    def range_prefix(self, prefix: str, *, endpoint: str | None = None) -> dict:
        encoded = prefix.encode()
        return self.api(
            endpoint or self.endpoint(),
            "/v3/kv/range",
            payload={
                "key": base64.b64encode(encoded).decode(),
                "range_end": base64.b64encode(prefix_end(encoded)).decode(),
            },
        )

    def snapshot(self, phase: str, names: list[str] | None = None) -> dict:
        names = names or list(self.nodes)
        endpoint = self.endpoint(set(self.nodes) - set(names))
        row = {
            "phase": phase,
            "statuses": self.statuses(names),
            "members": self.api(endpoint, "/v3/cluster/member/list", payload={}),
            "namespace": self.range_prefix(
                f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/", endpoint=endpoint
            ),
            "observed_unix_ns": time.time_ns(),
        }
        self.validate_snapshot(row, require_namespace=phase != "bootstrap")
        atomic_json(self.output / f"snapshot-{phase}.json", row)
        self.log_event({"event": "snapshot", "phase": phase})
        return row

    def validate_snapshot(self, snapshot: dict, *, require_namespace: bool) -> None:
        statuses = snapshot["statuses"]
        cluster_ids = {row["header"]["cluster_id"] for row in statuses.values()}
        leaders = {row["leader"] for row in statuses.values()}
        if len(cluster_ids) != 1 or len(leaders) != 1 or next(iter(leaders)) in ("0", 0):
            raise RuntimeError(f"inconsistent etcd status: {statuses}")
        cluster_id = next(iter(cluster_ids))
        if self.cluster_id is None:
            self.cluster_id = cluster_id
        elif cluster_id != self.cluster_id:
            raise RuntimeError(f"etcd cluster identity changed: {cluster_id} != {self.cluster_id}")
        revisions = [int(row["header"]["revision"]) for row in statuses.values()]
        if min(revisions) < max(self.last_revision, self.last_write_revision):
            raise RuntimeError(f"etcd revision regressed: {revisions} < {self.last_revision}")
        self.last_revision = min(revisions)
        members = snapshot["members"].get("members", [])
        expected = {node["name"]: node for node in self.settings["nodes"]}
        observed = {member["name"]: member for member in members}
        if set(observed) != set(expected):
            raise RuntimeError(f"etcd member names differ: {observed}")
        for name, node in expected.items():
            member = observed[name]
            if member["peerURLs"] != [node["peer_url"]]:
                raise RuntimeError(f"{name} peer URL differs: {member}")
            if member["clientURLs"] != [node["client_url"]]:
                raise RuntimeError(f"{name} client URL differs: {member}")
            if member.get("isLearner", False):
                raise RuntimeError(f"{name} is unexpectedly a learner: {member}")
            if statuses[name]["header"]["member_id"] != member["ID"]:
                raise RuntimeError(
                    f"{name} status/member identity differs: {statuses[name]}, {member}"
                )
        status_ids = {row["header"]["member_id"] for row in statuses.values()}
        member_ids = {member["ID"] for member in members}
        if status_ids != member_ids:
            raise RuntimeError(f"member IDs differ: {status_ids} != {member_ids}")
        ids_by_name = {name: observed[name]["ID"] for name in observed}
        if self.initial_member_ids is None:
            self.initial_member_ids = ids_by_name
        elif ids_by_name != self.initial_member_ids:
            raise RuntimeError(
                f"retained-data member identity changed: {ids_by_name} != {self.initial_member_ids}"
            )

        prefix = f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/"
        decoded_rows = {}
        namespace_header = snapshot["namespace"].get("header")
        if not isinstance(namespace_header, dict):
            raise RuntimeError("OrbitKV namespace response lacks an etcd header")
        namespace_cluster = str(namespace_header.get("cluster_id", ""))
        namespace_revision = int(namespace_header.get("revision", 0))
        if namespace_cluster != self.cluster_id or namespace_revision < max(
            self.last_revision, self.last_write_revision
        ):
            raise RuntimeError(
                f"OrbitKV namespace header differs: {namespace_header}, "
                f"cluster={self.cluster_id}, revision={self.last_revision}"
            )
        for row in snapshot["namespace"].get("kvs", []):
            key = base64.b64decode(row["key"]).decode()
            if key in decoded_rows:
                raise RuntimeError(f"duplicate OrbitKV namespace key: {key}")
            decoded_rows[key] = {
                "value": base64.b64decode(row.get("value", "")).decode(errors="strict"),
                "lease": int(row.get("lease", 0)),
                "create_revision": int(row.get("create_revision", 0)),
                "mod_revision": int(row.get("mod_revision", 0)),
            }
            record = decoded_rows[key]
            if not (0 < record["create_revision"] <= record["mod_revision"] <= namespace_revision):
                raise RuntimeError(f"invalid namespace revision evidence: {key}: {record}")
        decoded = {key: row["value"] for key, row in decoded_rows.items()}
        snapshot["decoded_namespace"] = decoded
        snapshot["decoded_namespace_records"] = decoded_rows
        if require_namespace:
            active = set(self.required_orbitkv_members)
            allowed = active | {
                "a100-consumer",
                "metadata-pressure",
                "query-source",
                "cold-observer",
            }
            required = {f"{prefix}format"} | {
                f"{prefix}{kind}/{node}" for kind in ("members", "epochs") for node in active
            }
            allowed_keys = (
                {f"{prefix}format"}
                | {f"{prefix}epochs/{node}" for node in allowed}
                | {f"{prefix}members/{node}" for node in active}
            )
            observed_keys = set(decoded)
            if not required <= observed_keys:
                raise RuntimeError(
                    f"required OrbitKV namespace keys missing: {required - observed_keys}"
                )
            if not observed_keys <= allowed_keys:
                raise RuntimeError(
                    f"unexpected OrbitKV namespace keys: {observed_keys - allowed_keys}"
                )
            format_record = decoded_rows[f"{prefix}format"]
            cluster_format = json.loads(format_record["value"])
            format_uuid = uuid.UUID(cluster_format["cluster_uuid"])
            if (
                format_record["lease"] != 0
                or set(cluster_format) != {"protocol", "cluster_uuid"}
                or cluster_format["protocol"] != self.settings["inventory_protocol"]
                or format_uuid.int == 0
            ):
                raise RuntimeError(f"invalid cluster format: {format_record}")
            if self.format_uuid is None:
                self.format_uuid = format_uuid
            elif format_uuid != self.format_uuid:
                raise RuntimeError(
                    f"persistent cluster format changed: {format_uuid} != {self.format_uuid}"
                )
            for node in allowed:
                epoch_key = f"{prefix}epochs/{node}"
                member_key = f"{prefix}members/{node}"
                if epoch_key not in decoded_rows:
                    continue
                epoch_record = decoded_rows[epoch_key]
                try:
                    epoch = int(epoch_record["value"])
                except ValueError as error:
                    raise RuntimeError(f"invalid epoch: {epoch_record}") from error
                if epoch <= 0 or epoch_record["lease"] != 0:
                    raise RuntimeError(f"invalid persistent epoch: {epoch_record}")
                if member_key not in decoded_rows:
                    continue
                member_record = decoded_rows[member_key]
                member = json.loads(member_record["value"])
                endpoint = member.get("owner", {}).get("endpoint", "")
                try:
                    endpoint_host, endpoint_port = endpoint.rsplit(":", 1)
                    endpoint_address = ipaddress.ip_address(endpoint_host.strip("[]"))
                    endpoint_port = int(endpoint_port)
                except (ValueError, AttributeError) as error:
                    raise RuntimeError(f"invalid member endpoint: {endpoint}") from error
                if (
                    member_record["lease"] == 0
                    or set(member) != {"node_id", "epoch", "owner", "protocol"}
                    or member["node_id"] != node
                    or member["epoch"] != epoch
                    or member["protocol"] != self.settings["inventory_protocol"]
                    or set(member["owner"]) != {"endpoint", "incarnation"}
                    or not endpoint_address.is_loopback
                    or not 0 < endpoint_port <= 65535
                    or uuid.UUID(member["owner"]["incarnation"]).int == 0
                ):
                    raise RuntimeError(f"invalid member record: {member_record}")
                expected_endpoint = self.expected_orbitkv_endpoints.get(node)
                if expected_endpoint is not None and endpoint != expected_endpoint:
                    raise RuntimeError(
                        f"{node} runtime endpoint differs: {endpoint} != {expected_endpoint}"
                    )
                incarnation = member["owner"]["incarnation"]
                previous_epoch = self.node_epochs.get(node)
                previous_incarnation = self.node_incarnations.get(node)
                if previous_epoch is not None and epoch < previous_epoch:
                    raise RuntimeError(f"node epoch regressed: {node}: {epoch} < {previous_epoch}")
                if (
                    previous_epoch is not None
                    and epoch == previous_epoch
                    and incarnation != previous_incarnation
                ):
                    raise RuntimeError(f"node incarnation changed without a new epoch: {node}")
                if (
                    previous_epoch is not None
                    and epoch > previous_epoch
                    and incarnation == previous_incarnation
                ):
                    raise RuntimeError(f"node epoch advanced while reusing its incarnation: {node}")
                if incarnation in self.retired_incarnations.get(node, set()):
                    raise RuntimeError(
                        f"retired node incarnation reappeared: {node}: {incarnation}"
                    )
                if (
                    previous_epoch is not None
                    and epoch > previous_epoch
                    and previous_incarnation is not None
                ):
                    self.retired_incarnations.setdefault(node, set()).add(previous_incarnation)
                self.node_epochs[node] = epoch
                self.node_incarnations[node] = incarnation

    def validate_agent_member_evidence(self, node: str, evidence: dict, snapshot: dict) -> dict:
        prefix = f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/"
        key = f"{prefix}members/{node}"
        header = evidence.get("header", {})
        record = evidence.get("record", {})
        snapshot_header = snapshot.get("namespace", {}).get("header", {})
        snapshot_record = snapshot.get("decoded_namespace_records", {}).get(key, {})
        try:
            header_revision = int(header.get("revision", 0))
            snapshot_revision = int(snapshot_header.get("revision", 0))
            record_projection = {
                "lease": int(record.get("lease", 0)),
                "create_revision": int(record.get("create_revision", 0)),
                "mod_revision": int(record.get("mod_revision", 0)),
            }
            snapshot_projection = {
                name: int(snapshot_record.get(name, 0))
                for name in ("lease", "create_revision", "mod_revision")
            }
            snapshot_member = json.loads(snapshot_record["value"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{node} agent member evidence is malformed: {evidence}") from error
        if (
            evidence.get("node") != node
            or evidence.get("requested_key") != key
            or evidence.get("endpoint")
            not in {entry["client_url"] for entry in self.settings["nodes"]}
            or str(header.get("cluster_id", "")) != self.cluster_id
            or not 0 < header_revision <= snapshot_revision
            or record.get("key") != key
            or record_projection != snapshot_projection
            or record_projection["lease"] <= 0
            or record.get("member") != snapshot_member
        ):
            raise RuntimeError(
                f"{node} agent/controller member evidence differs: "
                f"{evidence}, {snapshot_record}, {snapshot_header}"
            )
        return snapshot_member

    def validate_member_absence(self, node: str, absence: dict, previous: dict | None) -> None:
        key = f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/members/{node}"
        responses = absence.get("responses", {})
        expected_endpoints = {entry["client_url"] for entry in self.settings["nodes"]}
        minimum_revision = (
            int(previous.get("record", {}).get("mod_revision", 0)) if previous is not None else 1
        )
        if (
            absence.get("node") != node
            or absence.get("requested_key") != key
            or set(responses) != expected_endpoints
            or any(
                str(response.get("header", {}).get("cluster_id", "")) != self.cluster_id
                or int(response.get("header", {}).get("revision", 0)) < minimum_revision
                or response.get("records") != 0
                for response in responses.values()
            )
            or absence.get("previous_record") != previous
            or absence.get("member_absent") is not True
        ):
            raise RuntimeError(
                f"{node} authenticated member absence differs: {absence}, {previous}"
            )

    def phase(self, name: str) -> None:
        if self.resource_locks:
            self.state.setdefault("resource_lock_checks", []).append(
                {"phase": name, "boundary": "start", "locks": self.verify_resource_locks()}
            )
        if name == "bootstrap" and self.phase_deadline is not None:
            budget = self.settings["budgets"]["listener_preflight_seconds"]
        else:
            budget = (
                self.settings["budgets"]["listener_preflight_seconds"]
                if name == "bootstrap"
                else self.settings["budgets"]["phase_seconds"]
            )
            self.phase_deadline = time.monotonic() + budget
        self.state["current"] = name
        self.state["phase_deadline_mono"] = self.phase_deadline
        self.state["phase_budget_seconds"] = budget
        self.save()
        self.log_event({"event": "phase_start", "phase": name})

    def complete_phase(self, name: str, evidence: dict) -> None:
        self.remaining()
        if self.resource_locks:
            self.state.setdefault("resource_lock_checks", []).append(
                {"phase": name, "boundary": "complete", "locks": self.verify_resource_locks()}
            )
        self.validate_phase_evidence(name, evidence)
        row = {"phase": name, "status": "passed", "evidence": evidence}
        self.state["phases"].append(row)
        self.state["current"] = None
        self.state["phase_deadline_mono"] = None
        self.phase_deadline = None
        self.save()
        self.log_event({"event": "phase_complete", "phase": name})

    def validate_phase_evidence(self, name: str, evidence: dict) -> None:
        if not evidence:
            raise RuntimeError(f"{name}: empty phase evidence")
        if "local" in evidence:
            payload = evidence["local"].get("payload_sha256")
            if self.local_payload_sha256 is None or payload != self.local_payload_sha256:
                raise RuntimeError(f"{name}: local payload oracle differs: {payload}")
        if "snapshot" in evidence and not evidence["snapshot"].get("decoded_namespace"):
            raise RuntimeError(f"{name}: empty OrbitKV namespace evidence")
        if name == "orbitkv_consumer":
            if not evidence["converged"]["metadata"]["index"]["registration_valid"]:
                raise RuntimeError("consumer registration is not valid")
            if not evidence["oracle"].get("decoded_namespace"):
                raise RuntimeError("consumer etcd namespace is empty")
            reported_members = evidence.get("members", {})
            required_members = {
                "a100-consumer",
                "metadata-pressure",
                "query-source",
            }
            if set(reported_members) != required_members:
                raise RuntimeError(
                    f"consumer pre-fault member evidence is incomplete: {reported_members}"
                )
            authenticated_members = {
                node: self.validate_agent_member_evidence(
                    node, reported_members[node], evidence["oracle"]
                )
                for node in required_members
            }
            pressure_member = authenticated_members["metadata-pressure"]
            if pressure_member["owner"] != evidence.get("pressure_ready", {}).get("owner"):
                raise RuntimeError(
                    f"pressure readiness/member identity differs: "
                    f"{evidence.get('pressure_ready')}, {pressure_member}"
                )
            self.prefault_member_evidence = json.loads(json.dumps(reported_members))
        elif name == "leader_stop_rejoin":
            if int(evidence["new_term"]) <= int(evidence["old_term"]):
                raise RuntimeError("leader phase did not increase raft term")
        elif name == "quorum_loss":
            if not evidence.get("rejection"):
                raise RuntimeError("quorum loss lacks rejected write evidence")
            if evidence["fenced"]["metadata"]["index"]["registration_valid"]:
                raise RuntimeError("remote admission remained valid without quorum")
            if not evidence["pressure_result"].get("final_records"):
                raise RuntimeError("quorum-loss inventory result is empty")
        elif name == "quorum_restore":
            if not evidence["recovered"]["metadata"]["index"]["registration_valid"]:
                raise RuntimeError("consumer did not recover registration")
            recovery = evidence["recovery"]
            restarted = {"a100-consumer", "metadata-pressure", "query-source"}
            if (
                set(recovery.get("old_members", {})) != restarted
                or set(recovery.get("old_member_absence", {})) != restarted
                or set(recovery.get("members", {})) != restarted
                or recovery.get("old_members") != self.prefault_member_evidence
            ):
                raise RuntimeError("quorum recovery lacks authenticated pre-fault/member evidence")
            authenticated_current = {}
            for node, old_evidence in recovery.get("old_members", {}).items():
                absence = recovery.get("old_member_absence", {}).get(node, {})
                self.validate_member_absence(node, absence, old_evidence)
                current = self.validate_agent_member_evidence(
                    node, recovery["members"][node], evidence["snapshot"]
                )
                authenticated_current[node] = current
                old_member = old_evidence["record"]["member"]
                if (
                    int(current["epoch"]) <= int(old_member["epoch"])
                    or current["owner"]["incarnation"] == old_member["owner"]["incarnation"]
                ):
                    raise RuntimeError(
                        f"{node} restart identity did not advance: "
                        f"{old_member}, {absence}, {current}"
                    )
            pressure_member = authenticated_current["metadata-pressure"]
            pressure_ready = recovery.get("pressure", {}).get("ready", {})
            installed_pressure = self.owner(
                evidence["recovered"], pressure_member["owner"]["incarnation"]
            )
            if (
                pressure_ready.get("owner") != pressure_member["owner"]
                or installed_pressure is None
                or not installed_pressure.get("fresh")
            ):
                raise RuntimeError(
                    f"recovered pressure identity is not bound: "
                    f"{pressure_ready}, {pressure_member}, {installed_pressure}"
                )
        elif name == "lease_revoke_replacement":
            if int(evidence["new_member"]["epoch"]) <= int(
                evidence["revoke"]["old_member"]["epoch"]
            ):
                raise RuntimeError("replacement node epoch did not increase")
            if not evidence["replacement_result"].get("final_records"):
                raise RuntimeError("replacement inventory result is empty")
            if evidence["replacement_installed"]["pressure_poll"] is not None:
                raise RuntimeError("replacement pressure source exited unexpectedly")
            replacement_member = self.validate_agent_member_evidence(
                "metadata-pressure",
                evidence.get("replacement", {}).get("member", {}),
                evidence.get("replacement_snapshot", {}),
            )
            replacement_ready = (
                evidence.get("replacement", {}).get("replacement", {}).get("ready", {})
            )
            installed_replacement = self.owner(
                evidence["replacement_installed"],
                replacement_member["owner"]["incarnation"],
            )
            if (
                replacement_member != evidence["new_member"]
                or replacement_ready.get("owner") != replacement_member["owner"]
                or installed_replacement is None
                or not installed_replacement.get("fresh")
            ):
                raise RuntimeError(
                    f"replacement pressure identity is not bound: "
                    f"{replacement_ready}, {replacement_member}, {installed_replacement}"
                )
            live_query = evidence.get("live_source_query", {})
            stale_query = evidence.get("stale_source_query", {})
            if (
                live_query.get("expected_hits") != 1
                or live_query.get("observed_hits") != 1
                or live_query.get("lease_observed") is not True
                or stale_query.get("expected_hits") != 0
                or stale_query.get("observed_hits") != 0
                or stale_query.get("lease_observed") is not False
                or live_query.get("hash") != stale_query.get("hash")
                or live_query.get("source_endpoint")
                != evidence.get("query_source_member", {}).get("owner", {}).get("endpoint")
                or not evidence.get("query_source_revoke", {}).get("member_absent")
                or stale_query.get("source_endpoint") != live_query.get("source_endpoint")
                or live_query.get("query_reserved_bytes_after") != 0
                or stale_query.get("query_reserved_bytes_after") != 0
                or not evidence.get("stale_query_preparation", {}).get("local_resaved")
            ):
                raise RuntimeError(
                    f"live/stale source query control differs: {live_query}, {stale_query}"
                )
        elif name == "cold_observer_final_oracle":
            if not evidence.get("keys") or not evidence["final"].get("decoded_namespace"):
                raise RuntimeError("final etcd oracle is empty")
            if not evidence.get("storage"):
                raise RuntimeError("final etcd storage evidence is empty")
            if not evidence["cold"]["metadata"]["index"]["registration_valid"]:
                raise RuntimeError("cold observer registration is invalid")
            cold_member = self.validate_agent_member_evidence(
                "cold-observer",
                evidence["cold"].get("member_evidence", {}),
                evidence["cold"].get("controller_snapshot", {}),
            )
            cold_runtime = evidence["cold"].get("runtime_owner", {})
            controller_runtime = evidence["cold"].get("controller_runtime_identity", {})
            self.validate_member_absence(
                "cold-observer",
                evidence["cold"].get("preexisting_member_absence", {}),
                None,
            )
            self.validate_member_absence(
                "cold-observer",
                evidence["cold"].get("member_absence", {}),
                evidence["cold"].get("member_evidence", {}),
            )
            if (
                cold_member["owner"]["endpoint"] != self.expected_orbitkv_endpoints["cold-observer"]
                or cold_runtime.get("node") != "cold-observer"
                or cold_runtime.get("status") != "running"
                or cold_runtime.get("peer_endpoint") != cold_member["owner"]["endpoint"]
                or cold_runtime.get("process_live") is not True
                or not isinstance(cold_runtime.get("pid"), int)
                or not isinstance(cold_runtime.get("start_ticks"), str)
                or not cold_runtime["start_ticks"]
                or controller_runtime.get("pid") != cold_runtime.get("pid")
                or controller_runtime.get("start_ticks") != cold_runtime.get("start_ticks")
                or controller_runtime.get("listener")
                != [
                    "127.0.0.1",
                    self.settings["orbitkv_runtime"]["observer_peer_port"],
                    cold_runtime.get("pid"),
                ]
            ):
                raise RuntimeError(
                    f"cold observer member/runtime identity differs: {cold_member}, {cold_runtime}"
                )
            cold_termination = evidence["cold"].get("termination", {})
            cold_cleanup = evidence["cold"].get("manager_cleanup", {})
            if (
                cold_termination.get("status") != "stopped"
                or cold_termination.get("returncode") != 0
                or not cold_termination.get("sigterm_requested")
                or cold_cleanup.get("status") not in ("exited", "already_released")
                or cold_cleanup.get("returncode", 0) not in (None, 0)
            ):
                raise RuntimeError(
                    f"cold observer exit/cleanup is incomplete: {cold_termination}, {cold_cleanup}"
                )
            consumer_cleanup = evidence["cleanup"]["consumer"]
            query_source_cleanup = consumer_cleanup.get("query_source", {})
            drain = consumer_cleanup.get("manager_drain", {})
            required_drain = {
                "orbitkv_query_reserved_bytes",
                "orbitkv_query_reserved_bytes_by_phase",
                "orbitkv_transfer_completion_outstanding",
            }
            if (
                consumer_cleanup["manager_exit"] != 0
                or consumer_cleanup["pressure"]["status"] != "stopped"
                or consumer_cleanup["pressure"]["returncode"] != 0
                or consumer_cleanup["manager_cleanup"]["status"] != "already_released"
                or query_source_cleanup.get("status") != "stopped"
                or query_source_cleanup.get("termination", {}).get("status") != "stopped"
                or query_source_cleanup.get("manager_cleanup", {}).get("status")
                != "already_released"
                or "orbitkv_transfer_lock_active" not in query_source_cleanup.get("drain", {})
                or query_source_cleanup["drain"]["orbitkv_transfer_lock_active"] != 0
                or not required_drain <= drain.keys()
                or any(drain[name] != 0 for name in required_drain)
            ):
                raise RuntimeError(f"consumer cleanup is incomplete: {consumer_cleanup}")
            for node, cleanup in evidence["cleanup"]["nodes"].items():
                if (
                    cleanup["status"]
                    not in (
                        "stopped",
                        "terminated_by_sigterm",
                        "already_stopped",
                    )
                    or (
                        cleanup["status"] == "terminated_by_sigterm"
                        and (cleanup.get("returncode") != -15 or not cleanup.get("sigterm_sent"))
                    )
                    or (
                        cleanup["status"] != "terminated_by_sigterm"
                        and cleanup.get("returncode") not in (None, 0)
                    )
                ):
                    raise RuntimeError(f"{node} cleanup is incomplete: {cleanup}")
            if evidence["postflight"]["compute"]:
                raise RuntimeError("postflight compute process remains")

    @staticmethod
    def owner(snapshot: dict, incarnation: str) -> dict | None:
        return next((row for row in snapshot["owners"] if row["owner"] == incarnation), None)

    def validate_owner_watermarks(self, snapshot: dict) -> None:
        for owner in snapshot.get("owners", []):
            incarnation = owner["owner"]
            sequence = int(owner["applied_sequence"])
            previous = self.owner_watermarks.get(incarnation, 0)
            if sequence < previous:
                raise RuntimeError(
                    f"owner watermark regressed: {incarnation}: {sequence} < {previous}"
                )
            self.owner_watermarks[incarnation] = sequence

    def wait_consumer(
        self,
        label: str,
        predicate,
        *,
        timeout: float = 30,
    ) -> dict:
        deadline = time.monotonic() + min(timeout, self.remaining())
        attempt = 0
        last = None
        while time.monotonic() < deadline:
            last = self.consumer_request(f"snapshot_{label}_{attempt:03d}")
            self.validate_owner_watermarks(last)
            if predicate(last):
                return last
            attempt += 1
            time.sleep(0.2)
        raise TimeoutError(f"consumer condition {label} failed: {last}")

    def revoke_member_lease(self, node: str, old_owner: str) -> dict:
        prefix = f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/members/{node}"
        result = self.range_prefix(prefix)
        rows = result.get("kvs", [])
        if len(rows) != 1:
            raise RuntimeError(f"expected one {node} member: {result}")
        lease = rows[0].get("lease")
        if lease in (None, "0", 0):
            raise RuntimeError(f"missing {node} lease: {rows[0]}")
        old_member = json.loads(base64.b64decode(rows[0]["value"]))
        response = self.api(self.endpoint(), "/v3/lease/revoke", payload={"ID": lease})
        stale = self.wait_consumer(
            "lease_revoked",
            lambda snap: (owner := self.owner(snap, old_owner)) is None or not owner["fresh"],
            timeout=30,
        )
        remaining = self.range_prefix(prefix)
        if remaining.get("kvs"):
            raise RuntimeError(f"revoked {node} member key remained: {remaining}")
        return {
            "lease": lease,
            "old_member": old_member,
            "response": response,
            "stale": stale,
            "member_absent": True,
        }

    def orbitkv_member(self, node: str) -> dict:
        key = f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/members/{node}"
        result = self.range_prefix(key)
        rows = result.get("kvs", [])
        if len(rows) != 1:
            raise RuntimeError(f"expected one {node} member: {result}")
        return json.loads(base64.b64decode(rows[0]["value"]))

    def storage_evidence(self) -> dict:
        evidence = {}
        for name, node in self.nodes.items():
            result = self.host_command(
                node,
                [
                    "du",
                    "-sb",
                    str(self.root / name / "data"),
                    str(self.root / name / "wal"),
                ],
            )
            status = self.remote_json(node, self.root / name / "agent-status.json")
            evidence[name] = {"du": result.stdout.splitlines(), "agent": status}
        return evidence

    def external_owner_evidence(self, owner: dict) -> dict:
        node = self.nodes[owner["node"]]
        path = Path(owner["status_path"])
        present = self.host_command(node, ["test", "-f", str(path)], check=False)
        if present.returncode != 0:
            if owner.get("required", False):
                raise RuntimeError(f"required resource owner status is absent: {owner}")
            return {**owner, "status": "absent"}
        status = self.remote_json(node, path)
        state = str(status.get("state", "")).upper()
        if state not in owner["terminal_states"]:
            raise RuntimeError(f"external resource owner is not terminal: {status}")
        pid = status.get("controller_pid")
        ticks = status.get("controller_start_ticks")
        if not isinstance(pid, int) or not isinstance(ticks, str) or not ticks:
            raise RuntimeError(f"external resource owner lacks PID identity: {status}")
        process = self.host_command(
            node,
            [
                "python3",
                "-c",
                (
                    "from pathlib import Path; import sys; p=Path('/proc/'+sys.argv[1]+'/stat'); "
                    "sys.exit(1 if p.exists() and p.read_text().rsplit(')', 1)[1].split()[19] == sys.argv[2] else 0)"
                ),
                str(pid),
                ticks,
            ],
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"terminal resource owner process is still present: {status}")
        return {**owner, "status": status, "process_absent": True}

    def resource_gate(self) -> dict:
        gate = self.settings["resource_gate"]
        a100 = self.nodes["s2-a100"]
        external_owners = [self.external_owner_evidence(owner) for owner in gate["external_owners"]]
        gpu = self.host_command(
            a100,
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
        ).stdout.strip()
        memory = int(gpu.split(",")[0].strip())
        compute = self.host_command(
            a100,
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
        ).stdout.strip()
        if memory != gate["required_gpu_memory_mib"] or compute:
            raise RuntimeError(f"A100 is not exclusive: gpu={gpu!r}, compute={compute!r}")

        for node in self.settings["nodes"]:
            input_digest = self.host_command(
                node,
                ["sha256sum", str(self.root / "inputs/SHA256SUMS")],
            ).stdout.split()[0]
            if input_digest != self.approved_input_manifest_sha256:
                raise RuntimeError(f"{node['name']} input manifest differs: {input_digest}")
            self.host_command(
                node,
                [
                    "python3",
                    "-c",
                    (
                        "import os,subprocess,sys; os.chdir(sys.argv[1]); "
                        "raise SystemExit(subprocess.run(['sha256sum','--check','--strict','--quiet','SHA256SUMS']).returncode)"
                    ),
                    str(self.root / "inputs"),
                ],
            )
            process_rows = self.host_command(node, ["ps", "-eo", "pid=,args="]).stdout
            allowed_pids = {self.resource_locks[node["name"]]["pid"]}
            if node["ssh"] is None:
                allowed_pids.add(self.state["controller_pid"])
            conflicting = [
                line
                for line in process_rows.splitlines()
                if int(line.split(None, 1)[0]) not in allowed_pids
                if any(
                    pattern in line
                    for pattern in self.settings["resource_gate"]["process_patterns"]
                )
                and "resource_lock_agent.py" not in line
            ]
            if conflicting:
                raise RuntimeError(
                    f"{node['name']} has conflicting current process owners: {conflicting}"
                )
            for component in ("frozen",):
                self.host_command(
                    node,
                    [
                        "python3",
                        "-c",
                        (
                            "import os,subprocess,sys; os.chdir(sys.argv[1]); "
                            "raise SystemExit(subprocess.run(['sha256sum','--check','--strict','--quiet','SHA256SUMS']).returncode)"
                        ),
                        str(self.root / component),
                    ],
                )
            ports = {
                self.settings["etcd"]["client_port"],
                self.settings["etcd"]["peer_port"],
            }
            self.require_ports_without_listeners(
                node,
                ports,
                context=f"{node['name']} etcd preflight",
            )

        runtime = self.settings["orbitkv_runtime"]
        runtime_root = Path(runtime["source_root"])
        projection_manifest = runtime_root / "projection/SHA256SUMS"
        projection_digest = self.host_command(
            a100, ["sha256sum", str(projection_manifest)]
        ).stdout.split()[0]
        if projection_digest != runtime["projection_manifest_sha256"]:
            raise RuntimeError(f"runtime projection manifest differs: {projection_digest}")
        self.host_command(
            a100,
            [
                "python3",
                "-c",
                (
                    "import os,subprocess,sys; os.chdir(sys.argv[1]); "
                    "raise SystemExit(subprocess.run(['sha256sum','--check','--strict','--quiet','SHA256SUMS']).returncode)"
                ),
                str(runtime_root / "projection"),
            ],
        )
        installed = runtime_root / "runtime-venv/lib/python3.11/site-packages"
        application_hashes = self.host_command(
            a100,
            [
                "sha256sum",
                str(runtime_root / "frozen/bin/orbitkv-cache-manager"),
                str(runtime_root / "frozen/bin/server-tests"),
                str(installed / "orbitkv/orbitkv.cpython-311-x86_64-linux-gnu.so"),
                str(runtime_root / "frozen/native/libtent_shared.so"),
            ],
        ).stdout
        observed_hashes = [line.split()[0] for line in application_hashes.splitlines()]
        expected_hashes = [
            runtime["manager_sha256"],
            runtime["server_tests_sha256"],
            runtime["extension_sha256"],
            runtime["tent_sha256"],
        ]
        if observed_hashes != expected_hashes:
            raise RuntimeError(
                f"A100 runtime hashes differ: {observed_hashes} != {expected_hashes}"
            )
        import_record = self.host_command(
            a100,
            [
                "env",
                f"PYTHONPATH={installed}:{runtime_root / 'projection'}",
                f"LD_LIBRARY_PATH={runtime_root / 'frozen/native'}",
                str(runtime_root / "runtime-venv/bin/python"),
                "-c",
                (
                    "import hashlib,json,pathlib,orbitkv,orbitkv.orbitkv as native; "
                    "p=pathlib.Path(native.__file__).resolve(); "
                    "print(json.dumps({'package':str(pathlib.Path(orbitkv.__file__).resolve()),"
                    "'extension':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))"
                ),
            ],
        )
        imported = json.loads(import_record.stdout)
        if (
            imported["sha256"] != runtime["extension_sha256"]
            or not imported["package"].startswith(f"{installed}/orbitkv/")
            or not imported["extension"].startswith(f"{installed}/orbitkv/")
        ):
            raise RuntimeError(f"installed import mismatch: {imported}")
        manager_ports = {
            runtime["manager_peer_port"],
            runtime["manager_http_port"],
            runtime["observer_peer_port"],
            runtime["observer_http_port"],
            runtime["query_source_peer_port"],
            runtime["query_source_http_port"],
        }
        self.require_ports_without_listeners(
            a100,
            manager_ports,
            context="A100 Manager preflight",
        )
        result = {
            "external_owners": external_owners,
            "resource_locks": dict(self.resource_locks),
            "gpu": gpu,
            "compute": compute,
            "input_manifest_sha256": self.approved_input_manifest_sha256,
            "projection_manifest_sha256": projection_digest,
            "process_patterns": self.settings["resource_gate"]["process_patterns"],
            "runtime_hashes": observed_hashes,
            "import": imported,
        }
        atomic_json(self.output / "resource-gate.json", result)
        return result

    def wait_process_absent(self, node: dict, record: dict, *, cleanup: bool = True) -> None:
        deadline = time.monotonic() + min(30, self.remaining(cleanup=cleanup))
        while time.monotonic() < deadline:
            self.remaining(cleanup=cleanup)
            result = self.host_command(
                node,
                [
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path; import sys; p=Path('/proc/'+sys.argv[1]+'/stat'); "
                        "sys.exit(1 if p.exists() and p.read_text().rsplit(')', 1)[1].split()[19] == sys.argv[2] else 0)"
                    ),
                    str(record["pid"]),
                    record["start_ticks"],
                ],
                check=False,
                cleanup=cleanup,
            )
            if result.returncode == 0:
                return
            time.sleep(0.1)
        raise TimeoutError(f"owned process remained: {node['name']} {record}")

    def postflight(self) -> dict:
        for name, record in self.state["node_agents"].items():
            self.wait_process_absent(self.nodes[name], record)
        if self.state["consumer"] is not None:
            self.wait_process_absent(self.nodes["s2-a100"], self.state["consumer"])
        port_checks = {}
        for name, node in self.nodes.items():
            expected_ports = {
                self.settings["etcd"]["client_port"],
                self.settings["etcd"]["peer_port"],
            }
            self.require_ports_without_listeners(
                node,
                expected_ports,
                context=f"{name} etcd postflight",
                cleanup=True,
            )
            port_checks[name] = {
                "status": "no_listener",
                "address": str(ipaddress.ip_address(node["ipv6"])),
                "ports": sorted(expected_ports),
            }
        a100 = self.nodes["s2-a100"]
        gpu = self.host_command(
            a100,
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cleanup=True,
        ).stdout.strip()
        compute = self.host_command(
            a100,
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            cleanup=True,
        ).stdout.strip()
        if int(gpu.split(",")[0].strip()) != 0 or compute:
            raise RuntimeError(f"postflight A100 not empty: gpu={gpu!r}, compute={compute!r}")
        runtime = self.settings["orbitkv_runtime"]
        manager_ports = {
            runtime["manager_peer_port"],
            runtime["manager_http_port"],
            runtime["observer_peer_port"],
            runtime["observer_http_port"],
            runtime["query_source_peer_port"],
            runtime["query_source_http_port"],
        }
        self.require_ports_without_listeners(
            a100,
            manager_ports,
            context="A100 Manager postflight",
            cleanup=True,
        )
        socket_check = self.host_command(
            a100,
            [
                "python3",
                "-c",
                (
                    "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
                    "sys.exit(1 if p.exists() or p.is_symlink() else 0)"
                ),
                runtime["uds_root"],
            ],
            cleanup=True,
            check=False,
        )
        if socket_check.returncode != 0:
            raise RuntimeError("consumer UDS root remained")
        result = {
            "gpu": gpu,
            "compute": compute,
            "ports": port_checks,
            "uds_root_absent": True,
            "observed_unix_ns": time.time_ns(),
        }
        atomic_json(self.output / "postflight.json", result)
        return result

    @staticmethod
    def listener_port(line: str) -> int | None:
        fields = line.split()
        if len(fields) < 5:
            return None
        local = fields[3]
        try:
            _, port = local.rsplit(":", 1)
            return int(port)
        except ValueError:
            return None

    def require_ports_without_listeners(
        self,
        node: dict,
        ports: set[int],
        *,
        context: str,
        cleanup: bool = False,
    ) -> dict:
        output = self.host_command(
            node,
            ["ss", "-H", "-ltnp"],
            cleanup=cleanup,
        ).stdout
        conflicts = [line for line in output.splitlines() if self.listener_port(line) in ports]
        if conflicts:
            raise RuntimeError(f"{context} has live listeners: {conflicts}")
        return {"status": "no_listener", "ports": sorted(ports)}

    @staticmethod
    def listener_identity(line: str) -> tuple[str, int, int] | None:
        fields = line.split()
        if len(fields) < 5:
            return None
        local = fields[3]
        if local.startswith("[") and "]:" in local:
            address, _ = local[1:].rsplit("]:", 1)
        else:
            try:
                address, _ = local.rsplit(":", 1)
            except ValueError:
                return None
        match = re.search(r"\bpid=(\d+)\b", line)
        if match is None:
            return None
        parsed_port = Campaign.listener_port(line)
        if parsed_port is None:
            return None
        try:
            parsed_address = str(ipaddress.ip_address(address.split("%", 1)[0]))
        except ValueError:
            return None
        return parsed_address, parsed_port, int(match.group(1))

    def manager_listener_evidence(
        self,
        runtime_owner: dict,
        *,
        peer_port: int | None = None,
        label: str = "cold Manager",
    ) -> dict:
        node = self.nodes["s2-a100"]
        pid = runtime_owner.get("pid")
        ticks = runtime_owner.get("start_ticks")
        if not isinstance(pid, int) or not isinstance(ticks, str) or not ticks:
            raise RuntimeError(f"Manager runtime owner is incomplete: {runtime_owner}")
        identity = self.host_command(
            node,
            [
                "python3",
                "-c",
                (
                    "from pathlib import Path; import sys; p=Path('/proc/'+sys.argv[1]+'/stat'); "
                    "sys.exit(0 if p.exists() and p.read_text().rsplit(')', 1)[1].split()[19] == sys.argv[2] else 1)"
                ),
                str(pid),
                ticks,
            ],
            check=False,
        )
        if identity.returncode != 0:
            raise RuntimeError(f"Manager runtime owner is no longer live: {runtime_owner}")
        output = self.host_command(node, ["ss", "-H", "-ltnp"]).stdout
        listeners = [
            parsed
            for line in output.splitlines()
            if (parsed := self.listener_identity(line)) is not None
        ]
        expected = (
            "127.0.0.1",
            peer_port or self.settings["orbitkv_runtime"]["observer_peer_port"],
            pid,
        )
        if listeners.count(expected) != 1:
            raise RuntimeError(f"{label} exact listener identity differs: {expected}, {listeners}")
        return {
            "pid": pid,
            "start_ticks": ticks,
            "listener": list(expected),
            "raw": output.splitlines(),
        }

    def listener_evidence(self, names: list[str] | None = None) -> dict:
        result = {}
        for name in names or list(self.nodes):
            node = self.nodes[name]
            agent = self.remote_json(node, self.root / name / "agent-status.json")
            child = agent.get("child")
            if agent.get("state") != "running" or not isinstance(child, dict):
                raise RuntimeError(f"{name} etcd child is not running: {agent}")
            if not isinstance(child.get("start_ticks"), str) or not child["start_ticks"]:
                raise RuntimeError(f"{name} etcd child identity is incomplete: {child}")
            identity = self.host_command(
                node,
                [
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path; import sys; p=Path('/proc/'+sys.argv[1]+'/stat'); "
                        "sys.exit(0 if p.exists() and p.read_text().rsplit(')', 1)[1].split()[19] == sys.argv[2] else 1)"
                    ),
                    str(child["pid"]),
                    child["start_ticks"],
                ],
                check=False,
            )
            if identity.returncode != 0:
                raise RuntimeError(f"{name} etcd child identity is no longer live: {child}")
            output = self.host_command(node, ["ss", "-H", "-ltnp"]).stdout
            listeners = [
                parsed
                for line in output.splitlines()
                if (parsed := self.listener_identity(line)) is not None
            ]
            for port in (
                self.settings["etcd"]["client_port"],
                self.settings["etcd"]["peer_port"],
            ):
                expected = (str(ipaddress.ip_address(node["ipv6"])), port, child["pid"])
                matches = [listener for listener in listeners if listener == expected]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"{name} exact listener identity differs for {port}: "
                        f"expected={expected}, observed={listeners}"
                    )
            result[name] = {"child": child, "listeners": listeners, "raw": output.splitlines()}
        path = self.output / f"listeners-{self.listener_observation:03d}.json"
        self.listener_observation += 1
        atomic_json(path, result)
        return result

    def run(self) -> None:
        self.validate_launch_authorization()
        if self.state["controller_start_ticks"] is None:
            raise RuntimeError("controller process identity is unavailable")
        self.output.mkdir(exist_ok=False)
        lock = (self.root / ".campaign.lock").open("x")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.save()
        try:
            self.state["remote_inputs"] = self.verify_remote_inputs()
            self.save()
            self.state["resource_locks"] = self.acquire_resource_locks()
            self.save()
            self.phase_deadline = (
                self.started + self.settings["budgets"]["listener_preflight_seconds"]
            )
            self.state["current"] = "resource_and_listener_preflight"
            self.state["phase_deadline_mono"] = self.phase_deadline
            self.save()
            self.state["network_preflight"] = self.network_preflight()
            self.save()
            self.state["resource_gate"] = self.resource_gate()
            self.save()
            self.phase("bootstrap")
            for node in self.settings["nodes"]:
                self.launch_node(node)
            all_names = list(self.nodes)
            statuses = self.wait_cluster(all_names, timeout=60)
            listeners = self.listener_evidence()
            sentinel = self.put("/orbitkv-s2-three-host-ha/sentinel", "bootstrap")
            bootstrap = self.snapshot("bootstrap")
            if len(bootstrap["members"].get("members", [])) != 3:
                raise RuntimeError("etcd member list is not exactly three")
            self.complete_phase(
                "bootstrap",
                {"statuses": statuses, "listeners": listeners, "sentinel": sentinel},
            )

            self.phase("orbitkv_consumer")
            consumer = self.launch_consumer()
            pressure_owner = consumer["pressure"]["ready"]["owner"]
            old_owner = pressure_owner["incarnation"]
            self.expected_orbitkv_endpoints["metadata-pressure"] = pressure_owner["endpoint"]
            self.local_payload_sha256 = consumer["local"]["payload_sha256"]
            converged = self.wait_consumer(
                "baseline",
                lambda snap: (
                    snap["pressure_poll"] is None
                    and snap["metadata"]["index"]["registration_valid"]
                    and (owner := self.owner(snap, old_owner)) is not None
                    and owner["fresh"]
                ),
                timeout=60,
            )
            local = self.consumer_request("local_baseline")
            baseline_oracle = self.snapshot("orbitkv-consumer")
            self.complete_phase(
                "orbitkv_consumer",
                {
                    "old_owner": old_owner,
                    "pressure_ready": consumer["pressure"]["ready"],
                    "members": consumer["members"],
                    "converged": converged,
                    "local": local,
                    "oracle": baseline_oracle,
                },
            )

            self.phase("follower_stop_rejoin")
            statuses = self.wait_cluster(all_names)
            leader = self.leader(statuses)
            follower = next(name for name in all_names if name != leader)
            self.node_request(follower, "stop")
            survivors = [name for name in all_names if name != follower]
            self.wait_cluster(survivors)
            committed = self.put(
                "/orbitkv-s2-three-host-ha/follower-stop",
                "committed",
                endpoint=self.endpoint({follower}),
            )
            local = self.consumer_request("local_follower_stop")
            self.node_request(follower, "start")
            self.wait_cluster(all_names)
            listeners = self.listener_evidence()
            snapshot = self.snapshot("follower-rejoined")
            self.complete_phase(
                "follower_stop_rejoin",
                {
                    "follower": follower,
                    "committed": committed,
                    "local": local,
                    "listeners": listeners,
                    "snapshot": snapshot,
                },
            )

            self.phase("leader_stop_rejoin")
            before = self.wait_cluster(all_names)
            old_leader = self.leader(before)
            old_term = before[old_leader]["raftTerm"]
            self.node_request(old_leader, "stop")
            survivors = [name for name in all_names if name != old_leader]
            after = self.wait_cluster(survivors)
            new_leader = self.leader(after)
            if new_leader == old_leader:
                raise RuntimeError("leader did not change")
            if int(after[new_leader]["raftTerm"]) <= int(old_term):
                raise RuntimeError(
                    f"raft term did not increase: {old_term} -> {after[new_leader]['raftTerm']}"
                )
            committed = self.put(
                "/orbitkv-s2-three-host-ha/leader-stop",
                "committed",
                endpoint=self.endpoint({old_leader}),
            )
            local = self.consumer_request("local_leader_stop")
            self.node_request(old_leader, "start")
            self.wait_cluster(all_names)
            listeners = self.listener_evidence()
            snapshot = self.snapshot("leader-rejoined")
            self.complete_phase(
                "leader_stop_rejoin",
                {
                    "old_leader": old_leader,
                    "new_leader": new_leader,
                    "old_term": old_term,
                    "new_term": after[new_leader]["raftTerm"],
                    "committed": committed,
                    "local": local,
                    "listeners": listeners,
                    "snapshot": snapshot,
                },
            )

            self.phase("quorum_loss")
            statuses = self.wait_cluster(all_names)
            leader = self.leader(statuses)
            second = next(name for name in all_names if name != leader)
            stopped = [leader, second]
            for name in stopped:
                self.node_request(name, "stop")
            remaining = next(name for name in all_names if name not in stopped)
            try:
                self.put(
                    "/orbitkv-s2-three-host-ha/quorum-loss",
                    "must-not-commit",
                    endpoint=self.nodes[remaining]["client_url"],
                    timeout=5,
                )
            except requests.RequestException as error:
                rejection = repr(error)
            else:
                raise RuntimeError("write unexpectedly committed without quorum")
            fenced = self.wait_consumer(
                "quorum_loss",
                lambda snap: (
                    snap["pressure_poll"] is None
                    and not snap["metadata"]["index"]["registration_valid"]
                    and ((owner := self.owner(snap, old_owner)) is None or not owner["fresh"])
                ),
                timeout=60,
            )
            local = self.consumer_request("local_quorum_loss")
            pressure_result = self.consumer_request(
                "wait_pressure", timeout=self.settings["budgets"]["phase_seconds"]
            )
            self.complete_phase(
                "quorum_loss",
                {
                    "stopped": stopped,
                    "remaining": remaining,
                    "rejection": rejection,
                    "fenced": fenced,
                    "local": local,
                    "pressure_result": pressure_result,
                },
            )

            self.phase("quorum_restore")
            for name in stopped:
                self.node_request(name, "start")
            self.wait_cluster(all_names, timeout=60)
            listeners = self.listener_evidence()
            committed = self.put("/orbitkv-s2-three-host-ha/recovered", "committed")
            recovery = self.consumer_request("recover", timeout=120)
            recovered_pressure_owner = recovery["pressure"]["ready"]["owner"]
            recovered_owner = recovered_pressure_owner["incarnation"]
            self.expected_orbitkv_endpoints["metadata-pressure"] = recovered_pressure_owner[
                "endpoint"
            ]
            if recovered_owner == old_owner:
                raise RuntimeError("quorum recovery reused the old source incarnation")
            recovered = self.wait_consumer(
                "quorum_recovered",
                lambda snap: (
                    snap["pressure_poll"] is None
                    and snap["metadata"]["index"]["registration_valid"]
                    and (owner := self.owner(snap, recovered_owner)) is not None
                    and owner["fresh"]
                    and ((old := self.owner(snap, old_owner)) is None or not old["fresh"])
                ),
                timeout=60,
            )
            local = self.consumer_request("local_quorum_recovered")
            snapshot = self.snapshot("quorum-recovered")
            self.complete_phase(
                "quorum_restore",
                {
                    "committed": committed,
                    "recovery": recovery,
                    "old_owner": old_owner,
                    "recovered_owner": recovered_owner,
                    "recovered": recovered,
                    "local": local,
                    "listeners": listeners,
                    "snapshot": snapshot,
                },
            )

            self.phase("lease_revoke_replacement")
            recovered_pressure_result = self.consumer_request(
                "wait_pressure", timeout=self.settings["budgets"]["phase_seconds"]
            )
            query_source_member = self.orbitkv_member("query-source")
            query_source_owner = query_source_member["owner"]["incarnation"]
            query_source_installed = self.wait_consumer(
                "query_source_installed",
                lambda snap: (
                    (owner := self.owner(snap, query_source_owner)) is not None and owner["fresh"]
                ),
                timeout=60,
            )
            live_source_query = self.consumer_request("query_source_present")
            query_source_revoke = self.revoke_member_lease("query-source", query_source_owner)
            self.retired_incarnations.setdefault("query-source", set()).add(query_source_owner)
            self.required_orbitkv_members.remove("query-source")
            stale_query_preparation = self.consumer_request("prepare_query_source_absent")
            stale_source_query = self.consumer_request("query_source_absent")
            revoke = self.revoke_member_lease("metadata-pressure", recovered_owner)
            replacement = self.consumer_request("replacement", timeout=60)
            replacement_owner = replacement["replacement"]["ready"]["owner"]
            new_owner = replacement_owner["incarnation"]
            self.expected_orbitkv_endpoints["metadata-pressure"] = replacement_owner["endpoint"]
            if new_owner in (old_owner, recovered_owner):
                raise RuntimeError("replacement reused an old incarnation")
            replacement_converged = self.wait_consumer(
                "replacement",
                lambda snap: (
                    snap["pressure_poll"] is None
                    and (new := self.owner(snap, new_owner)) is not None
                    and new["fresh"]
                    and ((old := self.owner(snap, recovered_owner)) is None or not old["fresh"])
                ),
                timeout=60,
            )
            replacement_snapshot = self.snapshot("pressure-replacement")
            pressure_key = (
                f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/members/metadata-pressure"
            )
            new_member = json.loads(replacement_snapshot["decoded_namespace"][pressure_key])
            if int(new_member["epoch"]) <= int(revoke["old_member"]["epoch"]):
                raise RuntimeError(
                    f"pressure epoch did not increase: {revoke['old_member']} -> {new_member}"
                )
            replacement_result = self.consumer_request("wait_pressure", timeout=60)
            replacement_installed = self.wait_consumer(
                "replacement_final_watermark",
                lambda snap: (
                    snap["pressure_poll"] is None
                    and (source := self.owner(snap, new_owner)) is not None
                    and source["fresh"]
                    and source["applied_sequence"] == replacement_result["final_sequence"]
                    and source["records"] == len(replacement_result["final_records"])
                ),
                timeout=60,
            )
            final_records_sha256 = hashlib.sha256(
                json.dumps(
                    replacement_result["final_records"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            self.complete_phase(
                "lease_revoke_replacement",
                {
                    "first_owner": old_owner,
                    "old_owner": recovered_owner,
                    "new_owner": new_owner,
                    "pressure_result": recovered_pressure_result,
                    "live_source_query": live_source_query,
                    "query_source_member": query_source_member,
                    "query_source_installed": query_source_installed,
                    "query_source_revoke": query_source_revoke,
                    "stale_query_preparation": stale_query_preparation,
                    "revoke": revoke,
                    "stale_source_query": stale_source_query,
                    "replacement": replacement,
                    "replacement_snapshot": replacement_snapshot,
                    "replacement_converged": replacement_converged,
                    "new_member": new_member,
                    "replacement_result": replacement_result,
                    "replacement_installed": replacement_installed,
                    "final_records_sha256": final_records_sha256,
                },
            )

            self.phase("cold_observer_final_oracle")
            cold = self.consumer_request("cold_observer_start", timeout=120)
            cold["controller_runtime_identity"] = self.manager_listener_evidence(
                cold["runtime_owner"]
            )
            self.required_orbitkv_members.add("cold-observer")
            cold["controller_snapshot"] = self.snapshot("cold-observer-live")
            cold.update(self.consumer_request("cold_observer_stop", timeout=60))
            self.required_orbitkv_members.remove("cold-observer")
            consumer_final = self.consumer_request("snapshot_final_consumer")
            self.validate_owner_watermarks(cold)
            self.validate_owner_watermarks(consumer_final)
            if (
                cold["metadata"]["stream"]["scope_digest"]
                != consumer_final["metadata"]["stream"]["scope_digest"]
            ):
                raise RuntimeError("cold observer scope differs from active consumer")
            cold_source = self.owner(cold, new_owner)
            consumer_source = self.owner(consumer_final, new_owner)
            if (
                cold_source is None
                or consumer_source is None
                or not cold_source["fresh"]
                or not consumer_source["fresh"]
                or cold_source["view_id"] != consumer_source["view_id"]
                or cold_source["applied_sequence"] != consumer_source["applied_sequence"]
                or cold_source["records"] != consumer_source["records"]
            ):
                raise RuntimeError(
                    f"cold observer source differs: {cold_source} != {consumer_source}"
                )
            final = self.snapshot("final")
            keys = [
                base64.b64decode(row["key"]).decode() for row in final["namespace"].get("kvs", [])
            ]
            if any("/blocks/" in key or "/publishers/" in key for key in keys):
                raise RuntimeError(f"unexpected etcd block/cursor key: {keys}")
            local = self.consumer_request("local_final")
            storage = self.storage_evidence()
            final_evidence = {
                "cold": cold,
                "consumer": consumer_final,
                "final": final,
                "keys": keys,
                "local": local,
                "storage": storage,
            }

            cleanup = {
                "consumer": self.consumer_request("exit", timeout=120),
                "nodes": {},
            }
            prefix = f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/"
            deadline = time.monotonic() + min(30, self.remaining())
            while True:
                after_consumer = self.range_prefix(prefix)
                after_header = after_consumer.get("header", {})
                after_revision = int(after_header.get("revision", 0))
                if str(
                    after_header.get("cluster_id", "")
                ) != self.cluster_id or after_revision < max(
                    self.last_revision, self.last_write_revision
                ):
                    raise RuntimeError(f"post-consumer namespace header differs: {after_header}")
                decoded_after_records = {
                    base64.b64decode(row["key"]).decode(): {
                        "value": base64.b64decode(row.get("value", "")).decode(errors="strict"),
                        "lease": int(row.get("lease", 0)),
                        "create_revision": int(row.get("create_revision", 0)),
                        "mod_revision": int(row.get("mod_revision", 0)),
                    }
                    for row in after_consumer.get("kvs", [])
                }
                decoded_after = {key: row["value"] for key, row in decoded_after_records.items()}
                if not any("/members/" in key for key in decoded_after):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"OrbitKV member leases remained after consumer exit: {decoded_after}"
                    )
                time.sleep(0.1)
            for key, record in decoded_after_records.items():
                if not (0 < record["create_revision"] <= record["mod_revision"] <= after_revision):
                    raise RuntimeError(f"post-consumer namespace revision differs: {key}: {record}")
            expected_after = {
                f"{prefix}format",
                f"{prefix}epochs/a100-consumer",
                f"{prefix}epochs/metadata-pressure",
                f"{prefix}epochs/query-source",
                f"{prefix}epochs/cold-observer",
            }
            if set(decoded_after) != expected_after:
                raise RuntimeError(
                    f"post-consumer namespace differs: {set(decoded_after)} != {expected_after}"
                )
            final_format = json.loads(decoded_after_records[f"{prefix}format"]["value"])
            final_format_uuid = uuid.UUID(final_format["cluster_uuid"])
            if (
                decoded_after_records[f"{prefix}format"]["lease"] != 0
                or final_format.get("protocol") != self.settings["inventory_protocol"]
                or set(final_format) != {"protocol", "cluster_uuid"}
                or final_format_uuid.int == 0
                or final_format_uuid != self.format_uuid
            ):
                raise RuntimeError(f"post-consumer format is invalid: {final_format}")
            for key in expected_after - {f"{prefix}format"}:
                record = decoded_after_records[key]
                epoch = int(record["value"])
                node = key.rsplit("/", 1)[1]
                if record["lease"] != 0 or epoch <= 0 or epoch < self.node_epochs.get(node, 0):
                    raise RuntimeError(f"post-consumer epoch is invalid: {key}: {record}")
            cleanup["post_consumer_namespace"] = {
                "response": after_consumer,
                "decoded": decoded_after,
                "records": decoded_after_records,
            }
            for name in all_names:
                cleanup["nodes"][name] = self.node_request(name, "exit", timeout=60)
            postflight = self.postflight()
            self.complete_success(final_evidence, cleanup, postflight)
        except BaseException as error:
            timed_out = self.state["state"] == "timeout_quarantined"
            if not timed_out:
                self.state["state"] = "invalid"
            self.state["error"] = repr(error)
            self.save()
            self.log_event({"event": "failure", "error": repr(error)})
            cleanup = {"consumer": None, "nodes": {}}
            if timed_out:
                cleanup["skipped"] = (
                    "active host command identity and resource locks remain quarantined"
                )
                cleanup["resource_locks"] = {
                    name: {**record, "status": "retained_after_command_timeout"}
                    for name, record in self.resource_locks.items()
                }
                atomic_json(self.output / "failure-cleanup.json", cleanup)
                raise
            if self.network_probes:
                try:
                    cleanup["network_probes"] = self.stop_network_probes(cleanup=True)
                except BaseException as cleanup_error:
                    cleanup["network_probes"] = {"error": repr(cleanup_error)}
            if self.consumer_launched:
                try:
                    cleanup["consumer"] = self.consumer_request("exit", timeout=120, cleanup=True)
                except BaseException as cleanup_error:
                    cleanup["consumer"] = {"error": repr(cleanup_error)}
            for name in sorted(self.launched_nodes):
                try:
                    cleanup["nodes"][name] = self.node_request(
                        name, "exit", timeout=60, cleanup=True
                    )
                except BaseException as cleanup_error:
                    cleanup["nodes"][name] = {"error": repr(cleanup_error)}
            network_safe = all(
                record.get("status") == "stopped"
                for record in cleanup.get("network_probes", {}).values()
            ) and "error" not in cleanup.get("network_probes", {})
            consumer_safe = not self.consumer_launched or (
                isinstance(cleanup["consumer"], dict)
                and "error" not in cleanup["consumer"]
                and cleanup["consumer"].get("manager_exit") == 0
                and cleanup["consumer"].get("manager_cleanup", {}).get("status")
                == "already_released"
            )
            nodes_safe = all(
                record.get("status") in ("stopped", "terminated_by_sigterm", "already_stopped")
                for record in cleanup["nodes"].values()
            )
            if network_safe and consumer_safe and nodes_safe:
                cleanup["resource_locks"] = self.release_resource_locks()
            else:
                cleanup["resource_locks"] = {
                    name: {**record, "status": "retained_after_uncertain_cleanup"}
                    for name, record in self.resource_locks.items()
                }
                self.state["state"] = "timeout_quarantined"
            if any(
                record.get("status") != "released" for record in cleanup["resource_locks"].values()
            ):
                self.state["state"] = "timeout_quarantined"
            atomic_json(self.output / "failure-cleanup.json", cleanup)
            raise
        finally:
            self.state["elapsed_seconds"] = time.monotonic() - self.started
            self.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--approved-input-manifest-sha256", required=True)
    parser.add_argument("--launch-authorization", type=Path, required=True)
    parser.add_argument("--approved-launch-authorization-sha256", required=True)
    args = parser.parse_args()
    Campaign(
        args.root,
        args.approved_input_manifest_sha256,
        args.launch_authorization,
        args.approved_launch_authorization_sha256,
    ).run()


if __name__ == "__main__":
    main()
