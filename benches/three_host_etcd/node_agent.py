"""Own one three-host etcd child across reviewed stop/restart requests."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

try:
    from .environment import direct_network_environment, direct_network_environment_evidence
    from .integrity import verify_inputs, verify_process_runtime
except ImportError:
    from environment import direct_network_environment, direct_network_environment_evidence
    from integrity import verify_inputs, verify_process_runtime


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def listener_port(line: str) -> int | None:
    fields = line.split()
    if len(fields) < 5:
        return None
    try:
        return int(fields[3].rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def listener_conflicts(output: str, ports: set[int]) -> list[str]:
    return [line for line in output.splitlines() if listener_port(line) in ports]


class Owner:
    def __init__(
        self,
        root: Path,
        settings: dict,
        node: dict,
        approved_input_manifest_sha256: str,
    ):
        self.root = root
        self.settings = settings
        self.node = node
        self.approved_input_manifest_sha256 = approved_input_manifest_sha256
        self.environment = direct_network_environment(settings["network_environment"])
        self.node_root = root / node["name"]
        self.controls = self.node_root / "controls"
        self.responses = self.node_root / "responses"
        self.process: subprocess.Popen | None = None
        self.cleanup_deadline: float | None = None
        self.run_index = 0
        self.state = {
            "state": "starting",
            "agent_pid": os.getpid(),
            "agent_start_ticks": start_ticks(os.getpid()),
            "node": node,
            "child": None,
            "runs": [],
            "processed": [],
            "quarantined": [],
            "started_unix_ns": time.time_ns(),
            "network_environment": direct_network_environment_evidence(
                self.environment, settings["network_environment"]
            ),
        }

    @property
    def status_path(self) -> Path:
        return self.node_root / "agent-status.json"

    def save(self) -> None:
        if self.process is not None and self.process.poll() is None:
            status = Path(f"/proc/{self.process.pid}/status")
            stat = Path(f"/proc/{self.process.pid}/stat")
            try:
                fields = stat.read_text().split()
                memory = next(
                    line for line in status.read_text().splitlines() if line.startswith("VmRSS:")
                )
                self.state["child_resources"] = {
                    "pid": self.process.pid,
                    "user_ticks": fields[13],
                    "system_ticks": fields[14],
                    "rss": memory,
                }
            except (OSError, StopIteration):
                self.state["child_resources"] = {
                    "pid": self.process.pid,
                    "unavailable": True,
                }
        else:
            self.state["child_resources"] = None
        self.state["updated_unix_ns"] = time.time_ns()
        atomic_json(self.status_path, self.state)

    def command(self) -> list[str]:
        nodes = self.settings["nodes"]
        initial = ",".join(f"{item['name']}={item['peer_url']}" for item in nodes)
        etcd = self.settings["etcd"]
        return [
            str(self.root / "frozen/etcd"),
            "--name",
            self.node["name"],
            "--data-dir",
            str(self.node_root / "data"),
            "--wal-dir",
            str(self.node_root / "wal"),
            "--listen-client-urls",
            self.node["client_url"],
            "--advertise-client-urls",
            self.node["client_url"],
            "--listen-peer-urls",
            self.node["peer_url"],
            "--initial-advertise-peer-urls",
            self.node["peer_url"],
            "--initial-cluster",
            initial,
            "--initial-cluster-token",
            self.settings["cluster_token"],
            "--initial-cluster-state",
            "new",
            "--heartbeat-interval",
            str(etcd["heartbeat_ms"]),
            "--election-timeout",
            str(etcd["election_ms"]),
            "--log-level",
            "info",
        ]

    def preflight(self) -> None:
        verify_inputs(self.root, self.approved_input_manifest_sha256)
        self.state["process_runtime"] = verify_process_runtime(
            self.root, self.settings, self.node["name"]
        )
        if socket.gethostname() != self.node["hostname"]:
            raise RuntimeError(
                f"hostname mismatch: {socket.gethostname()} != {self.node['hostname']}"
            )
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if boot_id != self.node["boot_id"]:
            raise RuntimeError(f"boot ID mismatch: {boot_id} != {self.node['boot_id']}")
        binary = self.root / "frozen/etcd"
        if sha256(binary) != self.settings["etcd"]["sha256"]:
            raise RuntimeError("etcd binary hash mismatch")
        ports = {
            self.settings["etcd"]["client_port"],
            self.settings["etcd"]["peer_port"],
        }
        listeners = subprocess.run(
            ["ss", "-H", "-ltnp"],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if listeners.returncode != 0:
            raise RuntimeError(f"listener inspection failed: {listeners.stderr.strip()}")
        conflicts = listener_conflicts(listeners.stdout, ports)
        if conflicts:
            raise RuntimeError(f"etcd ports already have live listeners: {conflicts}")

    def start(self) -> dict:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("etcd child already running")
        command = self.command()
        log_path = self.node_root / f"etcd-run-{self.run_index:02d}.log"
        log = log_path.open("xb")
        process = subprocess.Popen(
            command,
            cwd=self.node_root,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        child = {
            "run_index": self.run_index,
            "pid": process.pid,
            "start_ticks": start_ticks(process.pid),
            "process_group": process.pid,
            "command": command,
            "log": str(log_path),
            "started_unix_ns": time.time_ns(),
            "network_environment": direct_network_environment_evidence(
                self.environment, self.settings["network_environment"]
            ),
        }
        self.run_index += 1
        self.process = process
        self.state["child"] = child
        self.state["runs"].append(child.copy())
        self.state["state"] = "running"
        self.save()
        return {"status": "started", **child}

    def stop(self, reason: str) -> dict:
        if self.process is None:
            if self.state["state"] == "child_exited":
                return {
                    "status": "preexisting_exit",
                    "reason": reason,
                    "evidence": self.state["runs"][-1]["unexpected_exit"],
                }
            return {"status": "already_stopped", "reason": reason, "returncode": None}
        process = self.process
        identity = self.state["child"].copy()
        sigterm_sent = False
        if process.poll() is not None:
            returncode = process.wait()
            status = "preexisting_exit"
        else:
            if start_ticks(process.pid) != identity.get("start_ticks"):
                record = {"status": "quarantined", "reason": "process_identity_changed", **identity}
                self.state["state"] = "timeout_quarantined"
                self.state["quarantined"].append(record)
                self.save()
                return record
            os.killpg(process.pid, signal.SIGTERM)
            sigterm_sent = True
            try:
                allowance = self.settings["budgets"]["normal_stop_seconds"]
                if self.cleanup_deadline is not None:
                    allowance = min(allowance, max(0.0, self.cleanup_deadline - time.monotonic()))
                returncode = process.wait(timeout=allowance)
            except subprocess.TimeoutExpired:
                record = {
                    "status": "quarantined",
                    "reason": reason,
                    **identity,
                    "observed_unix_ns": time.time_ns(),
                }
                self.state["state"] = "timeout_quarantined"
                self.state["quarantined"].append(record)
                self.save()
                return record
            if returncode == 0:
                status = "stopped"
            elif returncode == -signal.SIGTERM:
                status = "terminated_by_sigterm"
            else:
                status = "unexpected_exit"
        record = {
            "status": status,
            "reason": reason,
            **identity,
            "returncode": returncode,
            "sigterm_sent": sigterm_sent,
            "finished_unix_ns": time.time_ns(),
            "reaped": True,
        }
        self.process = None
        self.state["child"] = None
        self.state["runs"][-1]["stop"] = record
        self.state["state"] = (
            "stopped" if record["status"] in ("stopped", "terminated_by_sigterm") else "invalid"
        )
        self.save()
        return record

    def process_request(self, request: Path) -> bool:
        stem = request.name.removesuffix(".request")
        sequence, action = stem.split("-", 1)
        request_budget = json.loads(request.read_text() or "{}").get("cleanup_remaining_seconds")
        if request_budget is not None:
            if (
                type(request_budget) not in (int, float)
                or not math.isfinite(request_budget)
                or request_budget <= 0
                or request_budget > self.settings["budgets"]["cleanup_seconds"]
            ):
                raise RuntimeError("invalid remaining cleanup budget")
            local_deadline = time.monotonic() + request_budget
            self.cleanup_deadline = min(self.cleanup_deadline or local_deadline, local_deadline)
        response_path = self.responses / f"{stem}.json"
        if response_path.exists():
            return False
        if action == "stop":
            result = self.stop("reviewed_normal_fault")
        elif action == "start":
            result = self.start()
        elif action == "status":
            result = {"status": "observed", "state": self.state}
        elif action == "exit":
            result = self.stop("final_agent_exit")
            atomic_json(response_path, {"sequence": sequence, "action": action, **result})
            self.state["processed"].append(stem)
            if result["status"] == "quarantined":
                self.save()
                while True:
                    time.sleep(60)
            if result["status"] not in (
                "stopped",
                "terminated_by_sigterm",
                "already_stopped",
            ):
                self.state["state"] = "invalid"
                self.save()
                raise RuntimeError(f"final etcd cleanup failed: {result}")
            self.state["state"] = "exited"
            self.save()
            return True
        else:
            raise RuntimeError(f"unsupported action: {action}")
        atomic_json(response_path, {"sequence": sequence, "action": action, **result})
        self.state["processed"].append(stem)
        self.save()
        if result["status"] == "quarantined":
            while True:
                time.sleep(60)
        return False

    def run(self) -> None:
        self.node_root.mkdir(parents=True, exist_ok=False)
        self.controls.mkdir()
        self.responses.mkdir()
        lock = (self.node_root / ".agent.lock").open("x")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            self.preflight()
            self.start()
            while True:
                if self.process is not None:
                    code = self.process.poll()
                    if code is not None:
                        self.process.wait()
                        self.state["runs"][-1]["unexpected_exit"] = {
                            "returncode": code,
                            "observed_unix_ns": time.time_ns(),
                        }
                        self.process = None
                        self.state["child"] = None
                        self.state["state"] = "child_exited"
                        self.save()
                requests = sorted(self.controls.glob("*.request"))
                for request in requests:
                    stem = request.name.removesuffix(".request")
                    if stem in self.state["processed"]:
                        continue
                    try:
                        if self.process_request(request):
                            return
                    except BaseException as error:
                        sequence, action = stem.split("-", 1)
                        response = self.responses / f"{stem}.json"
                        if not response.exists():
                            atomic_json(
                                response,
                                {
                                    "sequence": sequence,
                                    "action": action,
                                    "status": "error",
                                    "error": repr(error),
                                },
                            )
                        self.state["processed"].append(stem)
                        raise
                time.sleep(0.1)
        except BaseException as error:
            self.state["state"] = "invalid"
            self.state["error"] = repr(error)
            cleanup = self.stop("node_agent_failure")
            self.state["failure_cleanup"] = cleanup
            if cleanup["status"] == "quarantined":
                self.state["state"] = "timeout_quarantined"
            self.save()
            atomic_json(self.node_root / "failure-cleanup.json", cleanup)
            if cleanup["status"] == "quarantined":
                while True:
                    time.sleep(60)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--approved-input-manifest-sha256", required=True)
    args = parser.parse_args()
    settings = json.loads((args.root / "inputs/settings.json").read_text())
    direct_network_environment_evidence(os.environ, settings["network_environment"])
    node = next(item for item in settings["nodes"] if item["name"] == args.node)
    Owner(args.root, settings, node, args.approved_input_manifest_sha256).run()


if __name__ == "__main__":
    main()
