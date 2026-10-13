"""CPU controls for the reusable three-host etcd process-fault harness."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import types
import uuid
from pathlib import Path

import pytest

from benches.three_host_etcd import campaign as campaign_module
from benches.three_host_etcd import consumer_agent as consumer_module
from benches.three_host_etcd import launch_agent as launch_module
from benches.three_host_etcd import node_agent as node_module
from benches.three_host_etcd import proxy_readiness as readiness_module
from benches.three_host_etcd.environment import (
    PROXY_ENVIRONMENT_VARIABLES,
    direct_network_environment,
    direct_network_environment_evidence,
    validated_manager_socket_paths,
)

INPUT_FILES = (
    "campaign.py",
    "consumer_agent.py",
    "environment.py",
    "integrity.py",
    "launch_agent.py",
    "network_probe_agent.py",
    "node_agent.py",
    "proxy_readiness.py",
    "resource_lock_agent.py",
    "validate_inputs.py",
)


def settings(lock_path: Path) -> dict:
    uds_root = Path("/tmp") / (
        "orbitkv-s2-etcd-" + hashlib.sha256(os.fsencode(lock_path)).hexdigest()[:12]
    )
    nodes = []
    for index, name in enumerate(("s2-h20", "s2-a100", "s2-cpu"), start=1):
        nodes.append(
            {
                "name": name,
                "host": name.removeprefix("s2-"),
                "hostname": os.uname().nodename,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "ipv6": "::1",
                "client_url": f"http://[::1]:{24000 + index}",
                "peer_url": f"http://[::1]:{25000 + index}",
                "ssh": None,
            }
        )
    return {
        "task_id": "three-host-etcd-unit",
        "source_commit": "unit-commit",
        "source_tree": "unit-tree",
        "formal_launch_authorized": False,
        "network_environment": {
            "unset": list(PROXY_ENVIRONMENT_VARIABLES),
            "no_proxy": "*",
        },
        "proxy_readiness": {"fake_proxy_url": "http://127.0.0.1:9"},
        "inventory_protocol": "orbitkv/inventory-stream/v4",
        "cluster_token": "unit-cluster",
        "orbitkv_cluster": "unit-cluster",
        "nodes": nodes,
        "etcd": {
            "client_port": 24000,
            "peer_port": 25000,
            "heartbeat_ms": 50,
            "election_ms": 500,
            "normal_stop_seconds": 0.1,
        },
        "budgets": {
            "cleanup_seconds": 1.0,
            "command_seconds": 0.2,
            "global_seconds": 10.0,
            "listener_preflight_seconds": 1.0,
            "normal_stop_seconds": 0.1,
            "phase_seconds": 1.0,
        },
        "resource_lock": {
            "path": str(lock_path),
            "resources": ["unit-test", f"uds-root-{uds_root.name}"],
        },
        "resource_gate": {
            "external_owners": [],
            "process_patterns": ["orbitkv-cache-manager", "server-tests"],
            "required_gpu_memory_mib": 0,
        },
        "orbitkv_runtime": {"uds_root": str(uds_root)},
    }


def test_direct_network_environment_removes_fake_proxies():
    policy = {
        "unset": list(PROXY_ENVIRONMENT_VARIABLES),
        "no_proxy": "*",
    }
    inherited = {
        **dict.fromkeys(PROXY_ENVIRONMENT_VARIABLES, "http://127.0.0.1:9"),
        "NO_PROXY": "localhost",
        "no_proxy": "localhost",
        "UNCHANGED": "yes",
    }
    environment = direct_network_environment(policy, source=inherited)
    evidence = direct_network_environment_evidence(environment, policy)

    assert all(name not in environment for name in PROXY_ENVIRONMENT_VARIABLES)
    assert environment["NO_PROXY"] == environment["no_proxy"] == "*"
    assert environment["UNCHANGED"] == "yes"
    assert evidence["proxy_variables_absent"] == list(PROXY_ENVIRONMENT_VARIABLES)
    with pytest.raises(RuntimeError, match="overrides controlled variables"):
        direct_network_environment(
            policy,
            source=inherited,
            updates={"HTTP_PROXY": "http://127.0.0.1:8"},
        )


def test_fake_proxy_breaks_inherited_http_and_direct_child_environment_recovers():
    listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    listener.bind(("::1", 0))
    listener.listen(1)
    listener.settimeout(3)
    port = listener.getsockname()[1]
    proxy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    proxy.bind(("127.0.0.1", 0))
    proxy_port = proxy.getsockname()[1]

    def serve():
        connection, _ = listener.accept()
        with connection:
            connection.recv(4096)
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
        listener.close()

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    policy = {
        "unset": list(PROXY_ENVIRONMENT_VARIABLES),
        "no_proxy": "*",
    }
    fake = dict(os.environ)
    for name in PROXY_ENVIRONMENT_VARIABLES:
        fake[name] = f"http://127.0.0.1:{proxy_port}"
    fake.update({"NO_PROXY": "", "no_proxy": ""})
    command = [
        sys.executable,
        "-c",
        (
            "import requests,sys; "
            "response=requests.get(sys.argv[1],timeout=2); "
            "response.raise_for_status(); print(response.status_code)"
        ),
        f"http://[::1]:{port}/v3/maintenance/status",
    ]
    blocked = subprocess.run(command, env=fake, capture_output=True, text=True, check=False)
    fixed = subprocess.run(
        command,
        env=direct_network_environment(policy, source=fake),
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    server.join(timeout=2)
    proxy.close()

    assert blocked.returncode != 0
    assert fixed.returncode == 0, fixed.stderr
    assert fixed.stdout.strip() == "200"
    assert not server.is_alive()


def test_campaign_and_consumer_sessions_ignore_fake_proxy_environment(tmp_path: Path, monkeypatch):
    for name in PROXY_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    root, config, digest = staged_root(tmp_path)

    campaign = campaign_module.Campaign(root, digest)
    consumer = consumer_module.Agent(root, config, digest)

    assert campaign.session.trust_env is False
    assert consumer.etcd_session.trust_env is False
    direct_network_environment_evidence(campaign.environment, config["network_environment"])
    direct_network_environment_evidence(consumer.environment, config["network_environment"])


def staged_root(tmp_path: Path, name: str = "stage") -> tuple[Path, dict, str]:
    root = tmp_path / name
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    source = Path(campaign_module.__file__).parent
    for filename in INPUT_FILES:
        shutil.copy2(source / filename, inputs / filename)
    config = settings(tmp_path / "exclusive.lock")
    config["orbitkv_runtime"].update(
        {
            "source_root": str(root),
            "manager_peer_port": 26001,
            "manager_http_port": 26002,
            "observer_peer_port": 26003,
            "observer_http_port": 26004,
            "query_source_peer_port": 26005,
            "query_source_http_port": 26006,
            "membership_ttl_seconds": 15,
            "metadata_namespace": "unit-namespace",
            "pressure_cadence_ms": 17,
            "pressure_keys": 32,
            "pressure_active_keys": 24,
            "pressure_window_shift": 4,
        }
    )
    (inputs / "settings.json").write_text(json.dumps(config, indent=2) + "\n")
    digest = write_manifest(root)
    return root, config, digest


def write_manifest(root: Path) -> str:
    inputs = root / "inputs"
    names = (*INPUT_FILES, "settings.json")
    manifest = "".join(
        f"{hashlib.sha256((inputs / filename).read_bytes()).hexdigest()}  {filename}\n"
        for filename in sorted(names)
    )
    (inputs / "SHA256SUMS").write_text(manifest)
    digest = hashlib.sha256((inputs / "SHA256SUMS").read_bytes()).hexdigest()
    return digest


def child(code: str) -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if "ready" in code:
        assert process.stdout.readline().strip() == "ready"
    return process


def attach(owner: node_module.Owner, process: subprocess.Popen) -> None:
    record = {
        "pid": process.pid,
        "start_ticks": node_module.start_ticks(process.pid),
        "process_group": process.pid,
    }
    owner.process = process
    owner.state["child"] = record
    owner.state["runs"] = [record.copy()]


def test_etcd_child_uses_direct_environment_under_fake_proxy(tmp_path: Path, monkeypatch):
    root, config, digest = staged_root(tmp_path)
    binary = root / "frozen/etcd"
    binary.parent.mkdir()
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "names = ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy')\n"
        "print(json.dumps({'present': [n for n in names if n in os.environ], "
        "'NO_PROXY': os.environ.get('NO_PROXY'), 'no_proxy': os.environ.get('no_proxy')}), "
        "flush=True)\n"
        "time.sleep(60)\n"
    )
    binary.chmod(0o755)
    for name in PROXY_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    owner = node_module.Owner(root, config, config["nodes"][0], digest)
    owner.node_root.mkdir()
    started = owner.start()
    log = Path(started["log"])
    deadline = time.monotonic() + 2
    while not log.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    child_environment = json.loads(log.read_text().splitlines()[0])
    assert child_environment == {"present": [], "NO_PROXY": "*", "no_proxy": "*"}
    assert started["network_environment"] == direct_network_environment_evidence(
        owner.environment, config["network_environment"]
    )
    stopped = owner.stop("test_complete")
    assert stopped["status"] == "stopped"
    assert stopped["returncode"] == 0


def test_remote_launcher_scrubs_fake_proxy_before_agent_start(tmp_path: Path, monkeypatch):
    root, config, digest = staged_root(tmp_path)
    captured = {}

    class Process:
        pid = os.getpid()

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return Process()

    for name in PROXY_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setattr(launch_module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_agent.py",
            "--root",
            str(root),
            "--kind",
            "resource-lock",
            "--node",
            "s2-h20",
            "--approved-input-manifest-sha256",
            digest,
        ],
    )

    launch_module.main()

    evidence = direct_network_environment_evidence(captured["env"], config["network_environment"])
    record = json.loads((root / "s2-h20.resource-lock.launcher.json").read_text())
    assert record["network_environment"] == evidence


@pytest.mark.parametrize("returncode", [0, 23])
def test_node_agent_rejects_preexisting_exit(tmp_path: Path, returncode: int):
    root, config, digest = staged_root(tmp_path)
    owner = node_module.Owner(root, config, config["nodes"][0], digest)
    owner.node_root.mkdir()
    process = child(f"raise SystemExit({returncode})")
    process.wait(timeout=2)
    attach(owner, process)
    result = owner.stop("reviewed_fault")
    assert result["status"] == "preexisting_exit"
    assert result["returncode"] == returncode
    assert not result["sigterm_sent"]


def test_node_agent_records_own_sigterm_exactly(tmp_path: Path):
    root, config, digest = staged_root(tmp_path)
    owner = node_module.Owner(root, config, config["nodes"][0], digest)
    owner.node_root.mkdir()
    process = child("import time; print('ready', flush=True); time.sleep(60)")
    attach(owner, process)
    result = owner.stop("reviewed_fault")
    assert result["status"] == "terminated_by_sigterm"
    assert result["returncode"] == -signal.SIGTERM
    assert result["sigterm_sent"]
    assert result["reaped"]


def test_node_agent_quarantines_sigterm_timeout_without_kill(tmp_path: Path):
    root, config, digest = staged_root(tmp_path)
    owner = node_module.Owner(root, config, config["nodes"][0], digest)
    owner.node_root.mkdir()
    process = child(
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(60)"
    )
    attach(owner, process)
    try:
        result = owner.stop("reviewed_fault")
        assert result["status"] == "quarantined"
        assert process.poll() is None
        assert owner.process is process
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        assert process.wait(timeout=2) == -signal.SIGKILL


def test_node_agent_listener_gate_uses_live_listeners_not_bind_availability():
    wildcard = 'LISTEN 0 4096 *:24000 *:* users:(("other",pid=42,fd=7))'
    ipv6 = 'LISTEN 0 4096 [::1]:25000 [::]:* users:(("other",pid=43,fd=8))'
    unrelated = 'LISTEN 0 4096 127.0.0.1:26000 0.0.0.0:* users:(("other",pid=44,fd=9))'
    output = "\n".join((wildcard, ipv6, unrelated))

    assert node_module.listener_port(wildcard) == 24000
    assert node_module.listener_port(ipv6) == 25000
    assert node_module.listener_conflicts(output, {24000, 25000}) == [wildcard, ipv6]
    assert node_module.listener_conflicts("", {24000, 25000}) == []


def test_manager_socket_paths_enforce_linux_path_limit(tmp_path: Path):
    token = hashlib.sha256(os.fsencode(tmp_path)).hexdigest()[:12]
    root = Path("/tmp") / f"orbitkv-s2-etcd-{token}"
    paths = validated_manager_socket_paths({"uds_root": str(root)})
    assert set(paths) == {"a100-consumer", "query-source", "cold-observer"}
    assert all(len(os.fsencode(path)) <= 107 for path in paths.values())

    too_long = Path("/tmp") / f"orbitkv-s2-etcd-{'x' * 100}"
    with pytest.raises(RuntimeError, match="exceeds 107 bytes"):
        validated_manager_socket_paths({"uds_root": str(too_long)})


def test_consumer_owns_probes_and_releases_short_uds_root(tmp_path: Path):
    agent = bare_agent(tmp_path)
    evidence = agent.prepare_uds_root()
    assert evidence["owner"]["path"] == str(agent.uds_root)
    assert all(record["bind"] == "passed" for record in evidence["paths"].values())
    assert all(not path.exists() for path in agent.manager_socket_paths.values())
    assert agent.release_uds_root()["status"] == "released"
    assert not agent.uds_root.exists()


def test_consumer_retains_changed_or_nonempty_uds_root(tmp_path: Path):
    agent = bare_agent(tmp_path)
    agent.prepare_uds_root()
    residue = agent.uds_root / "unknown"
    try:
        residue.write_text("do not remove")
        cleanup = agent.release_uds_root()
        assert cleanup["status"] == "retained_nonempty"
        assert cleanup["entries"] == ["unknown"]
        assert agent.uds_root.exists()
        residue.unlink()
        agent.uds_root.rmdir()
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        agent.uds_root.symlink_to(replacement, target_is_directory=True)
        assert agent.release_uds_root()["status"] == "ownership_changed"
        assert agent.uds_root.exists()
    finally:
        residue.unlink(missing_ok=True)
        agent.uds_root.unlink(missing_ok=True)


class ProcessManager:
    def __init__(self, process: subprocess.Popen, *, timeout: bool = False):
        self.process = process
        self.timeout = timeout
        self.http_port = 1

    def terminate_gracefully(self, *, timeout: float):
        os.killpg(self.process.pid, signal.SIGTERM)
        if self.timeout:
            with pytest.raises(subprocess.TimeoutExpired):
                self.process.wait(timeout=timeout)
            raise subprocess.TimeoutExpired("manager", timeout)
        started = time.monotonic()
        return self.process.wait(timeout=timeout), time.monotonic() - started

    def stop_without_kill(self, *, timeout: float):
        if self.process.poll() is None:
            if not self.timeout:
                os.killpg(self.process.pid, signal.SIGTERM)
                returncode = self.process.wait(timeout=timeout)
                return {
                    "status": "exited",
                    "pid": self.process.pid,
                    "start_ticks": None,
                    "sigterm_sent": True,
                    "forced_cleanup": False,
                    "reaped": True,
                    "returncode": returncode,
                }
            return {
                "status": "quarantined",
                "pid": self.process.pid,
                "start_ticks": consumer_module.start_ticks(self.process.pid),
            }
        return {"status": "already_released", "returncode": self.process.returncode}


def bare_agent(tmp_path: Path) -> consumer_module.Agent:
    root, config, digest = staged_root(tmp_path)
    agent = consumer_module.Agent(root, config, digest)
    agent.output.mkdir()
    agent.controls.mkdir()
    agent.responses.mkdir()
    return agent


def agent_member_evidence(
    agent: consumer_module.Agent,
    node: str,
    *,
    endpoint: str,
    epoch: int = 1,
    incarnation: str | None = None,
    cluster_id: str = "7",
    revision: int = 20,
) -> dict:
    key = f"/orbitkv/v2/{agent.settings['orbitkv_cluster']}/members/{node}"
    member = {
        "node_id": node,
        "epoch": epoch,
        "owner": {
            "endpoint": endpoint,
            "incarnation": incarnation or str(uuid.uuid4()),
        },
        "protocol": agent.settings["inventory_protocol"],
    }
    return {
        "node": node,
        "requested_key": key,
        "endpoint": agent.settings["nodes"][0]["client_url"],
        "header": {"cluster_id": cluster_id, "revision": revision},
        "record": {
            "key": key,
            "lease": 100,
            "create_revision": 1,
            "mod_revision": 2,
            "member": member,
        },
        "observed_unix_ns": time.time_ns(),
    }


def agent_member_absence(agent: consumer_module.Agent, node: str, previous: dict | None) -> dict:
    return {
        "node": node,
        "requested_key": (f"/orbitkv/v2/{agent.settings['orbitkv_cluster']}/members/{node}"),
        "responses": {
            entry["client_url"]: {
                "header": {"cluster_id": "7", "revision": 21},
                "records": 0,
            }
            for entry in agent.settings["nodes"]
        },
        "previous_record": previous,
        "member_absent": True,
        "observed_unix_ns": time.time_ns(),
    }


def attach_cold_runtime_owner(agent: consumer_module.Agent, process: subprocess.Popen) -> None:
    agent.state["manager_owners"].append(
        {
            "node": "cold-observer",
            "run": 0,
            "status": "running",
            "pid": process.pid,
            "start_ticks": consumer_module.start_ticks(process.pid),
        }
    )


def test_cold_observer_rejects_preexisting_exit(tmp_path: Path):
    agent = bare_agent(tmp_path)
    process = child("raise SystemExit(0)")
    process.wait(timeout=2)
    manager = ProcessManager(process)
    agent.manager_owners.append(manager)
    agent.start_manager = lambda **_: manager
    agent.metadata = lambda *_: {"index": {"registration_valid": True}}
    agent.owners = lambda *_: [{"owner": "source", "fresh": True}]
    agent.wait_member_absent = lambda node, previous: agent_member_absence(agent, node, previous)
    agent.wait_etcd_member = lambda _node: agent_member_evidence(
        agent, "cold-observer", endpoint="127.0.0.1:26003"
    )
    attach_cold_runtime_owner(agent, process)
    with pytest.raises(RuntimeError, match="runtime ownership differs"):
        agent.start_cold_observer()
    assert process.poll() is not None


def test_cold_observer_rejects_nonzero_requested_exit(tmp_path: Path):
    agent = bare_agent(tmp_path)
    process = child(
        "import signal,sys,time; signal.signal(signal.SIGTERM, lambda *_: sys.exit(23)); "
        "print('ready', flush=True); time.sleep(60)"
    )
    manager = ProcessManager(process)
    agent.manager_owners.append(manager)
    agent.start_manager = lambda **_: manager
    agent.metadata = lambda *_: {"index": {"registration_valid": True}}
    agent.owners = lambda *_: [{"owner": "source", "fresh": True}]
    agent.wait_member_absent = lambda node, previous: agent_member_absence(agent, node, previous)
    agent.wait_etcd_member = lambda _node: agent_member_evidence(
        agent, "cold-observer", endpoint="127.0.0.1:26003"
    )
    attach_cold_runtime_owner(agent, process)
    agent.start_cold_observer()
    with pytest.raises(RuntimeError, match="cleanup is unresolved"):
        agent.stop_cold_observer()
    assert process.poll() == 23


def test_cold_observer_timeout_retains_live_owner(tmp_path: Path):
    agent = bare_agent(tmp_path)
    process = child(
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(60)"
    )
    manager = ProcessManager(process, timeout=True)
    agent.manager_owners.append(manager)
    agent.start_manager = lambda **_: manager
    agent.metadata = lambda *_: {"index": {"registration_valid": True}}
    agent.owners = lambda *_: [{"owner": "source", "fresh": True}]
    agent.wait_member_absent = lambda node, previous: agent_member_absence(agent, node, previous)
    agent.wait_etcd_member = lambda _node: agent_member_evidence(
        agent, "cold-observer", endpoint="127.0.0.1:26003"
    )
    attach_cold_runtime_owner(agent, process)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            agent.start_cold_observer()
            agent.stop_cold_observer()
        assert manager in agent.manager_owners
        assert process.poll() is None
        cleanup = json.loads((agent.output / "cold-observer-cleanup.json").read_text())
        assert cleanup["status"] == "quarantined"
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def test_cold_observer_rejects_live_member_on_wrong_endpoint(tmp_path: Path):
    agent = bare_agent(tmp_path)
    clock = [0.0]
    agent.etcd_member = lambda _node: agent_member_evidence(
        agent, "cold-observer", endpoint="127.0.0.1:65535"
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(consumer_module.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(
            consumer_module.time,
            "sleep",
            lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        )
        with pytest.raises(TimeoutError, match="member did not appear"):
            agent.wait_etcd_member("cold-observer")


@pytest.mark.parametrize("mutation", ["wrong_cluster", "wrong_key", "zero_lease"])
def test_member_reader_authenticates_full_etcd_response(tmp_path: Path, mutation: str):
    agent = bare_agent(tmp_path)
    agent.etcd_cluster_id = "7"
    agent.etcd_revisions = {entry["client_url"]: 10 for entry in agent.settings["nodes"]}
    node = "cold-observer"
    key = f"/orbitkv/v2/{agent.settings['orbitkv_cluster']}/members/{node}"
    member = {
        "node_id": node,
        "epoch": 1,
        "owner": {
            "endpoint": "127.0.0.1:26003",
            "incarnation": str(uuid.uuid4()),
        },
        "protocol": agent.settings["inventory_protocol"],
    }
    response = {
        "header": {"cluster_id": "7", "revision": "20"},
        "kvs": [kv(key, member, 100)],
    }
    if mutation == "wrong_cluster":
        response["header"]["cluster_id"] = "8"
    elif mutation == "wrong_key":
        response["kvs"][0]["key"] = base64.b64encode(b"/unrelated/key").decode()
    else:
        response["kvs"][0]["lease"] = "0"
    with pytest.raises(RuntimeError):
        agent.decode_member_evidence(node, agent.settings["nodes"][0]["client_url"], response)


def test_member_absence_rejects_wrong_cluster_and_zero_revision(tmp_path: Path, monkeypatch):
    agent = bare_agent(tmp_path)
    agent.etcd_cluster_id = "7"
    agent.etcd_revisions = {entry["client_url"]: 10 for entry in agent.settings["nodes"]}
    response = types.SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"header": {"cluster_id": "8", "revision": "0"}, "kvs": []},
    )
    agent.etcd_session = types.SimpleNamespace(post=lambda *_args, **_kwargs: response)
    clock = [0.0]
    monkeypatch.setattr(consumer_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        consumer_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + 31 + seconds),
    )
    with pytest.raises(RuntimeError, match="response header|cluster identity"):
        agent.wait_member_absent("cold-observer", None)


def test_manager_start_exception_retains_real_child_owner(tmp_path: Path, monkeypatch):
    agent = bare_agent(tmp_path)
    created = []

    class StartingManager:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.process = None
            created.append(self)

        def start(self):
            self.process = child(
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); time.sleep(60)"
            )
            raise RuntimeError("injected readiness failure")

        def stop_without_kill(self, *, timeout: float):
            del timeout
            return {
                "status": "quarantined",
                "pid": self.process.pid,
                "start_ticks": consumer_module.start_ticks(self.process.pid),
            }

    monkeypatch.setattr(consumer_module, "CacheManagerProcess", StartingManager)
    try:
        with pytest.raises(RuntimeError, match="start raised"):
            agent.start_manager()
        assert created[0] in agent.manager_owners
        cleanup = agent.cleanup_partial("startup failure")
        assert cleanup["quarantined"]
        assert created[0] in agent.manager_owners
    finally:
        process = created[0].process
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def test_cleanup_error_retains_tensor_client_and_manager(tmp_path: Path):
    agent = bare_agent(tmp_path)
    process = child("import time; print('ready', flush=True); time.sleep(60)")

    class BrokenManager:
        def __init__(self):
            self.process = process

        def stop_without_kill(self, *, timeout: float):
            del timeout
            raise OSError("unresolved wait failure")

    class BrokenClient:
        def unregister_context(self, instance: str):
            del instance
            raise RuntimeError("native operation unresolved")

    manager = BrokenManager()
    agent.manager = manager
    agent.manager_owners.append(manager)
    agent.client = BrokenClient()
    agent.tensor = object()
    agent.expected = object()
    try:
        cleanup = agent.cleanup_partial("native failure")
        assert cleanup["quarantined"]
        assert not cleanup["tensor_released"]
        assert agent.client is not None
        assert manager in agent.manager_owners
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def test_query_source_client_cleanup_error_retains_both_tensors(tmp_path: Path):
    agent = bare_agent(tmp_path)
    agent.client = types.SimpleNamespace(
        unregister_context=lambda *_: (True, "ok"), close=lambda: None
    )

    class BrokenSourceClient:
        def unregister_context(self, *_args):
            raise RuntimeError("source native operation unresolved")

    agent.query_source_client = BrokenSourceClient()
    agent.tensor = object()
    agent.expected = object()
    agent.query_source_tensor = object()
    agent.query_source_hashes = [b"hash"]
    cleanup = agent.cleanup_partial("source client failure")
    assert cleanup["quarantined"]
    assert not cleanup["tensor_released"]
    assert agent.query_source_client is not None
    assert agent.tensor is not None and agent.query_source_tensor is not None


def test_quorum_recovery_uses_prefault_member_identities(tmp_path: Path, monkeypatch):
    agent = bare_agent(tmp_path)
    endpoints = {
        "a100-consumer": "127.0.0.1:26001",
        "metadata-pressure": "127.0.0.1:26002",
        "query-source": "127.0.0.1:26005",
    }
    old_members = {
        node: agent_member_evidence(
            agent,
            node,
            endpoint=endpoints[node],
            epoch=7,
            incarnation=str(uuid.UUID(int=index)),
        )
        for index, node in enumerate(endpoints, start=1)
    }
    agent.current_members = old_members
    agent.tensor = object()
    agent.expected = object()
    agent.client = types.SimpleNamespace(
        unregister_context=lambda *_: (True, "ok"), close=lambda: None
    )
    agent.manager = types.SimpleNamespace(process=types.SimpleNamespace(pid=os.getpid()))
    agent.stop_pressure = lambda *_: {"status": "stopped"}
    agent.stop_query_source = lambda *_args, **_kwargs: {"status": "stopped"}
    agent.drain_metrics = lambda: {"orbitkv_query_reserved_bytes": 0}
    agent._terminate_manager_gracefully = lambda *_args, **_kwargs: {
        "status": "stopped",
        "returncode": 0,
        "shutdown_seconds": 0.1,
    }
    agent._stop_manager_owner = lambda *_args, **_kwargs: {
        "status": "already_released",
        "returncode": 0,
    }
    absent = []
    agent.wait_member_absent = lambda node, previous: (
        absent.append((node, json.dumps(previous, sort_keys=True)))
        or agent_member_absence(agent, node, previous)
    )

    def start_local_cache():
        agent.manager = types.SimpleNamespace(process=types.SimpleNamespace(pid=os.getpid()))
        return {"status": "started"}

    agent.start_local_cache = start_local_cache
    agent.start_query_source = lambda: {"status": "started"}
    new_pressure_owner = {
        "endpoint": endpoints["metadata-pressure"],
        "incarnation": str(uuid.UUID(int=12)),
    }
    agent.start_pressure = lambda *_: {"ready": {"status": "ready", "owner": new_pressure_owner}}
    new_members = {
        node: agent_member_evidence(
            agent,
            node,
            endpoint=endpoints[node],
            epoch=8,
            incarnation=str(uuid.UUID(int=index + 10)),
            revision=22,
        )
        for index, node in enumerate(old_members, start=1)
    }
    agent.wait_etcd_member = lambda node, expected_owner=None: new_members[node]
    agent.etcd_member = lambda _node: (_ for _ in ()).throw(
        AssertionError("post-fault lookup must not reconstruct the old identity")
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: None, ipc_collect=lambda: None)
        ),
    )

    result = agent.recover_after_quorum()
    assert result["old_members"] == old_members
    assert result["members"] == new_members
    assert set(absent) == {
        (node, json.dumps(member, sort_keys=True)) for node, member in old_members.items()
    }


DRAIN_METRICS = (
    "orbitkv_query_reserved_bytes",
    "orbitkv_query_reserved_bytes_by_phase",
    "orbitkv_inflight_bytes",
    "orbitkv_transfer_lock_active",
    "orbitkv_transfer_reserved_bytes",
    "orbitkv_ssd_prefetch_inflight",
    "orbitkv_ssd_read_pinned_bytes",
    "orbitkv_ssd_write_inflight",
    "orbitkv_ssd_write_queue_pending",
    "orbitkv_transfer_completion_outstanding",
)


def drain_text(overrides=None, missing=None, omit_phase=None):
    overrides = overrides or {}
    lines = []
    for name in DRAIN_METRICS:
        if name == missing:
            continue
        if name == "orbitkv_query_reserved_bytes_by_phase":
            for phase in ["warming", "preloading", "prepared", "preparing", "ready", "restoring"]:
                if phase != omit_phase:
                    lines.append(f'{name}{{phase="{phase}"}} {overrides.get(phase, 0)}')
        else:
            lines.append(f"{name} {overrides.get(name, 0)}")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("missing", DRAIN_METRICS)
def test_drain_metrics_must_exist_and_be_zero(tmp_path, missing):
    agent = bare_agent(tmp_path)
    agent.manager = types.SimpleNamespace(http_port=1)
    agent.etcd_session = types.SimpleNamespace(
        get=lambda *_args, **_kw: types.SimpleNamespace(
            text=drain_text(), raise_for_status=lambda: None
        )
    )
    assert agent.drain_metrics()["orbitkv_query_reserved_bytes"] == 0
    agent.etcd_session.get = lambda *_args, **_kw: types.SimpleNamespace(
        text=drain_text(missing=missing), raise_for_status=lambda: None
    )
    with pytest.raises(RuntimeError, match="absent|incomplete"):
        agent.drain_metrics()


@pytest.mark.parametrize(
    "overrides,omit",
    [
        ({"warming": "NaN"}, None),
        ({"warming": "+Inf"}, None),
        ({"warming": 7, "ready": -7}, None),
        ({}, "ready"),
        ({"orbitkv_ssd_write_inflight": 1}, None),
    ],
)
def test_drain_rejects_each_phase_sample_before_aggregation(tmp_path, overrides, omit):
    agent = bare_agent(tmp_path)
    agent.manager = types.SimpleNamespace(http_port=1)
    agent.etcd_session = types.SimpleNamespace(
        get=lambda *_args, **_kw: types.SimpleNamespace(
            text=drain_text(overrides, omit_phase=omit), raise_for_status=lambda: None
        )
    )
    with pytest.raises(RuntimeError, match="invalid|incomplete"):
        agent.drain_metrics()


def test_readable_source_query_has_positive_then_stale_negative_control(
    tmp_path: Path, monkeypatch
):
    agent = bare_agent(tmp_path)
    agent.manager = types.SimpleNamespace(http_port=1)
    agent.query_source_hashes = [bytes(range(32))]
    released = []
    agent.client = types.SimpleNamespace(release=lambda lease: released.append(lease))

    def query(_client, _instance, _hashes, _request, expected, _managers):
        return types.SimpleNamespace(
            lease=b"lease" if expected == 1 else b"",
            num_hit_blocks=expected,
        )

    monkeypatch.setattr(consumer_module, "_query_ready", query)
    metrics = iter(
        [
            {"orbitkv_query_reserved_bytes": 0},
            {
                "orbitkv_query_reserved_bytes": 0,
                "orbitkv_remote_fetch_bytes_total": 4096,
            },
            {
                "orbitkv_query_reserved_bytes": 0,
                "orbitkv_remote_fetch_bytes_total": 4096,
            },
            {
                "orbitkv_query_reserved_bytes": 0,
                "orbitkv_remote_fetch_bytes_total": 4096,
            },
        ]
    )
    monkeypatch.setattr(consumer_module, "fetch_orbitkv_metrics", lambda _port: next(metrics))
    positive = agent.query_readable_source(expected_hits=1)
    negative = agent.query_readable_source(expected_hits=0)
    assert positive["observed_hits"] == 1 and positive["lease_observed"]
    assert negative["observed_hits"] == 0 and not negative["lease_observed"]
    assert positive["hash"] == negative["hash"]
    assert released == [b"lease"]


def test_readable_source_query_rejects_missing_remote_counter(tmp_path: Path, monkeypatch):
    agent = bare_agent(tmp_path)
    agent.manager = types.SimpleNamespace(http_port=1)
    agent.query_source_hashes = [bytes(range(32))]
    agent.client = types.SimpleNamespace(release=lambda _lease: None)
    monkeypatch.setattr(
        consumer_module,
        "_query_ready",
        lambda *_args: types.SimpleNamespace(lease=b"lease", num_hit_blocks=1),
    )
    monkeypatch.setattr(
        consumer_module,
        "fetch_orbitkv_metrics",
        lambda _port: {"orbitkv_query_reserved_bytes": 0},
    )
    with pytest.raises(RuntimeError, match="remote fetch byte counter is absent"):
        agent.query_readable_source(expected_hits=1)


def test_pressure_identity_is_durable_before_readiness_timeout(tmp_path: Path, monkeypatch):
    agent = bare_agent(tmp_path)
    agent.settings["orbitkv_runtime"].update(
        {
            "source_root": str(tmp_path),
            "pressure_cadence_ms": 17,
            "pressure_keys": 32,
            "pressure_active_keys": 24,
            "pressure_window_shift": 4,
        }
    )
    process = child(
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(60)"
    )
    clock = [0.0]
    monkeypatch.setattr(consumer_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(consumer_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        consumer_module.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    try:
        with pytest.raises(TimeoutError, match="did not become ready"):
            agent.start_pressure(10)
        assert agent.state["pressure_runs"][0]["pid"] == process.pid
        assert agent.state["pressure_runs"][0]["status"] == "starting"
        assert process.poll() is None
    finally:
        agent.pressure_log.close()
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def test_pressure_unrequested_zero_exit_is_not_a_planned_stop(tmp_path: Path):
    agent = bare_agent(tmp_path)
    artifact = agent.output / "pressure-00"
    artifact.mkdir()
    process = child("raise SystemExit(0)")
    process.wait(timeout=2)
    agent.pressure = process
    agent.pressure_output = artifact
    agent.pressure_log = (artifact / "pressure.log").open("w")
    with pytest.raises(RuntimeError, match="before stop request"):
        agent.stop_pressure("final_agent_exit")
    record = json.loads((artifact / "pressure-cleanup.json").read_text())
    assert record["status"] == "preexisting_exit"
    assert not record["stop_requested"]


def test_cleanup_budget_is_one_cumulative_deadline(tmp_path: Path, monkeypatch):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    campaign.output.mkdir()
    now = [100.0]
    monkeypatch.setattr(campaign_module.time, "monotonic", lambda: now[0])
    assert campaign.remaining(cleanup=True) == pytest.approx(1.0)
    now[0] += 0.6
    assert campaign.remaining(cleanup=True) == pytest.approx(0.4)
    now[0] += 0.5
    with pytest.raises(TimeoutError, match="cleanup watchdog"):
        campaign.remaining(cleanup=True)


def test_formal_run_requires_exact_independent_launch_authorization(tmp_path: Path):
    root, config, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    with pytest.raises(PermissionError, match="formal launch authorization"):
        campaign.validate_launch_authorization()
    with pytest.raises(PermissionError, match="formal launch authorization"):
        campaign.run()
    assert not campaign.output.exists()
    review_root = root / "review"
    review_root.mkdir()
    review = review_root / "REVIEW.md"
    review.write_text("ACCEPTED — exact freeze\n")
    verdict_path = review_root / "verdict.json"
    verdict_path.write_text(
        json.dumps(
            {
                "verdict": "ACCEPTED",
                "accepted": True,
                "candidate_commit": config["source_commit"],
                "candidate_tree": config["source_tree"],
                "input_manifest_sha256": digest,
            }
        )
        + "\n"
    )
    authorization = {
        "formal_launch_authorized": True,
        "formal_runs_started": 0,
        "task_id": config["task_id"],
        "source_commit": config.get("source_commit"),
        "source_tree": config.get("source_tree"),
        "input_manifest_sha256": digest,
        "launch_review_verdict": "ACCEPTED",
        "launch_review_path": str(review),
        "launch_review_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
        "launch_review_verdict_path": str(verdict_path),
        "launch_review_verdict_sha256": hashlib.sha256(verdict_path.read_bytes()).hexdigest(),
    }
    authorization_path = root / "LAUNCH-AUTHORIZATION.json"
    authorization_path.write_text(json.dumps(authorization, sort_keys=True) + "\n")
    authorization_sha256 = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
    campaign = campaign_module.Campaign(root, digest, authorization_path, authorization_sha256)
    campaign.validate_launch_authorization()

    authorization_path.write_text("{}\n")
    with pytest.raises(PermissionError, match="authorization identity differs"):
        campaign.validate_launch_authorization()

    review.write_text("CHANGES REQUIRED — rejected freeze\n")
    authorization["launch_review_sha256"] = hashlib.sha256(review.read_bytes()).hexdigest()
    authorization_path.write_text(json.dumps(authorization, sort_keys=True) + "\n")
    rejected_campaign = campaign_module.Campaign(
        root,
        digest,
        authorization_path,
        hashlib.sha256(authorization_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(PermissionError, match="review is not accepted"):
        rejected_campaign.validate_launch_authorization()


def test_launch_authorization_rejects_lexical_stage_escape(tmp_path: Path):
    root, config, digest = staged_root(tmp_path)
    outside_review = tmp_path / "outside-review.md"
    outside_review.write_text("ACCEPTED — escaped review\n")
    outside_verdict = tmp_path / "outside-verdict.json"
    outside_verdict.write_text(
        json.dumps(
            {
                "verdict": "ACCEPTED",
                "accepted": True,
                "candidate_commit": config["source_commit"],
                "candidate_tree": config["source_tree"],
                "input_manifest_sha256": digest,
            }
        )
        + "\n"
    )
    authorization = {
        "formal_launch_authorized": True,
        "formal_runs_started": 0,
        "task_id": config["task_id"],
        "source_commit": config["source_commit"],
        "source_tree": config["source_tree"],
        "input_manifest_sha256": digest,
        "launch_review_verdict": "ACCEPTED",
        "launch_review_path": str(root / ".." / outside_review.name),
        "launch_review_sha256": hashlib.sha256(outside_review.read_bytes()).hexdigest(),
        "launch_review_verdict_path": str(root / ".." / outside_verdict.name),
        "launch_review_verdict_sha256": hashlib.sha256(outside_verdict.read_bytes()).hexdigest(),
    }
    authorization_path = root / "LAUNCH-AUTHORIZATION.json"
    authorization_path.write_text(json.dumps(authorization) + "\n")
    campaign = campaign_module.Campaign(
        root,
        digest,
        authorization_path,
        hashlib.sha256(authorization_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(PermissionError, match="escapes the frozen stage"):
        campaign.run()
    assert not campaign.output.exists()


def test_success_validates_final_phase_before_releasing_locks(tmp_path: Path, monkeypatch):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    campaign.output.mkdir()
    campaign.state["current"] = "cold_observer_final_oracle"
    campaign.resource_locks.update(
        {
            name: {"status": "locked", "pid": index, "start_ticks": str(index)}
            for index, name in enumerate(campaign.nodes, start=1)
        }
    )
    events = []

    def verify():
        events.append("locks_live")
        return dict(campaign.resource_locks)

    def validate(name, evidence):
        assert name == "cold_observer_final_oracle"
        assert evidence["cleanup"] == {"consumer": {}, "nodes": {}}
        events.append("phase_validated")

    def release():
        events.append("locks_released")
        return {
            name: {
                "status": "released",
                "pid": record["pid"],
                "start_ticks": record["start_ticks"],
            }
            for name, record in campaign.resource_locks.items()
        }

    monkeypatch.setattr(campaign, "verify_resource_locks", verify)
    monkeypatch.setattr(campaign, "validate_phase_evidence", validate)
    monkeypatch.setattr(campaign, "release_resource_locks", release)
    cleanup = {"consumer": {}, "nodes": {}}
    campaign.complete_success({"oracle": True}, cleanup, {"compute": []})
    assert events == ["locks_live", "phase_validated", "locks_released"]
    assert campaign.state["state"] == "completed_pending_review"
    assert campaign.state["resource_lock_release"] == cleanup["resource_locks"]
    assert (
        json.loads((campaign.output / "resource-lock-release.json").read_text())
        == cleanup["resource_locks"]
    )


def kv(key: str, value: object, lease: int) -> dict:
    encoded = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    return {
        "key": base64.b64encode(key.encode()).decode(),
        "value": base64.b64encode(encoded.encode()).decode(),
        "lease": str(lease),
        "create_revision": "1",
        "mod_revision": "2",
    }


def valid_snapshot(campaign: campaign_module.Campaign) -> dict:
    ids = {name: str(index) for index, name in enumerate(campaign.nodes, start=1)}
    prefix = f"/orbitkv/v2/{campaign.settings['orbitkv_cluster']}/"
    rows = [
        kv(
            f"{prefix}format",
            {
                "protocol": campaign.settings["inventory_protocol"],
                "cluster_uuid": str(uuid.uuid4()),
            },
            0,
        )
    ]
    endpoints = {
        "a100-consumer": "127.0.0.1:26001",
        "metadata-pressure": "127.0.0.1:26002",
        "query-source": "127.0.0.1:26005",
    }
    for index, node in enumerate(endpoints, start=1):
        rows.extend(
            [
                kv(f"{prefix}epochs/{node}", str(index), 0),
                kv(
                    f"{prefix}members/{node}",
                    {
                        "node_id": node,
                        "epoch": index,
                        "owner": {
                            "endpoint": endpoints[node],
                            "incarnation": str(uuid.uuid4()),
                        },
                        "protocol": campaign.settings["inventory_protocol"],
                    },
                    100 + index,
                ),
            ]
        )
    return {
        "statuses": {
            name: {
                "header": {
                    "cluster_id": "7",
                    "member_id": ids[name],
                    "revision": "20",
                },
                "leader": "1",
            }
            for name in campaign.nodes
        },
        "members": {
            "members": [
                {
                    "name": name,
                    "ID": ids[name],
                    "peerURLs": [node["peer_url"]],
                    "clientURLs": [node["client_url"]],
                }
                for name, node in campaign.nodes.items()
            ]
        },
        "namespace": {
            "header": {"cluster_id": "7", "revision": "20"},
            "kvs": rows,
        },
    }


def member_evidence_from_snapshot(
    campaign: campaign_module.Campaign, snapshot: dict, node: str
) -> dict:
    key = f"/orbitkv/v2/{campaign.settings['orbitkv_cluster']}/members/{node}"
    record = snapshot["decoded_namespace_records"][key]
    return {
        "node": node,
        "requested_key": key,
        "endpoint": campaign.settings["nodes"][0]["client_url"],
        "header": {
            "cluster_id": snapshot["namespace"]["header"]["cluster_id"],
            "revision": int(snapshot["namespace"]["header"]["revision"]),
        },
        "record": {
            "key": key,
            "lease": record["lease"],
            "create_revision": record["create_revision"],
            "mod_revision": record["mod_revision"],
            "member": json.loads(record["value"]),
        },
        "observed_unix_ns": time.time_ns(),
    }


def member_absence_for_campaign(
    campaign: campaign_module.Campaign, node: str, previous: dict, revision: int
) -> dict:
    return {
        "node": node,
        "requested_key": (f"/orbitkv/v2/{campaign.settings['orbitkv_cluster']}/members/{node}"),
        "responses": {
            entry["client_url"]: {
                "header": {
                    "cluster_id": campaign.cluster_id,
                    "revision": revision,
                },
                "records": 0,
            }
            for entry in campaign.settings["nodes"]
        },
        "previous_record": previous,
        "member_absent": True,
        "observed_unix_ns": time.time_ns(),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "member_id",
        "extra_key",
        "leased_epoch",
        "member_value",
        "namespace_header",
        "record_revision",
        "invalid_endpoint",
        "invalid_pressure_port",
        "learner",
    ],
)
def test_snapshot_rejects_identity_namespace_and_lease_mutants(tmp_path: Path, mutation: str):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    snapshot = valid_snapshot(campaign)
    prefix = f"/orbitkv/v2/{campaign.settings['orbitkv_cluster']}/"
    if mutation == "member_id":
        snapshot["statuses"]["s2-a100"]["header"]["member_id"] = "3"
    elif mutation == "extra_key":
        snapshot["namespace"]["kvs"].append(kv(f"{prefix}unexpected", "bad", 0))
    elif mutation == "leased_epoch":
        next(
            row
            for row in snapshot["namespace"]["kvs"]
            if base64.b64decode(row["key"]).decode().endswith("epochs/a100-consumer")
        )["lease"] = "99"
    elif mutation == "member_value":
        row = next(
            row
            for row in snapshot["namespace"]["kvs"]
            if base64.b64decode(row["key"]).decode().endswith("members/a100-consumer")
        )
        value = json.loads(base64.b64decode(row["value"]))
        value["node_id"] = "wrong"
        row["value"] = base64.b64encode(json.dumps(value).encode()).decode()
    elif mutation == "namespace_header":
        snapshot["namespace"]["header"]["cluster_id"] = "99"
    elif mutation == "record_revision":
        snapshot["namespace"]["kvs"][0]["create_revision"] = "999"
    elif mutation == "invalid_endpoint":
        row = next(
            row
            for row in snapshot["namespace"]["kvs"]
            if base64.b64decode(row["key"]).decode().endswith("members/a100-consumer")
        )
        value = json.loads(base64.b64decode(row["value"]))
        value["owner"]["endpoint"] = "not-an-endpoint"
        row["value"] = base64.b64encode(json.dumps(value).encode()).decode()
    elif mutation == "invalid_pressure_port":
        row = next(
            row
            for row in snapshot["namespace"]["kvs"]
            if base64.b64decode(row["key"]).decode().endswith("members/metadata-pressure")
        )
        value = json.loads(base64.b64decode(row["value"]))
        value["owner"]["endpoint"] = "127.0.0.1:65536"
        row["value"] = base64.b64encode(json.dumps(value).encode()).decode()
    else:
        snapshot["members"]["members"][0]["isLearner"] = True
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        campaign.validate_snapshot(snapshot, require_namespace=True)


def set_member_identity(snapshot: dict, node: str, epoch: int, incarnation: str) -> None:
    for row in snapshot["namespace"]["kvs"]:
        key = base64.b64decode(row["key"]).decode()
        if key.endswith(f"epochs/{node}"):
            row["value"] = base64.b64encode(str(epoch).encode()).decode()
        elif key.endswith(f"members/{node}"):
            value = json.loads(base64.b64decode(row["value"]))
            value["epoch"] = epoch
            value["owner"]["incarnation"] = incarnation
            row["value"] = base64.b64encode(json.dumps(value).encode()).decode()


@pytest.mark.parametrize(
    "mutation", ["epoch_regression", "same_epoch_new_inc", "new_epoch_old_inc"]
)
@pytest.mark.parametrize("node", ["a100-consumer", "metadata-pressure", "query-source"])
def test_snapshot_requires_monotonic_epoch_and_new_restart_incarnation(
    tmp_path: Path, mutation: str, node: str
):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    baseline = valid_snapshot(campaign)
    set_member_identity(baseline, node, 10, str(uuid.UUID(int=10)))
    campaign.validate_snapshot(baseline, require_namespace=True)
    candidate = json.loads(json.dumps(baseline))
    if mutation == "epoch_regression":
        set_member_identity(candidate, node, 1, str(uuid.UUID(int=11)))
    elif mutation == "same_epoch_new_inc":
        set_member_identity(candidate, node, 10, str(uuid.UUID(int=11)))
    else:
        set_member_identity(candidate, node, 11, str(uuid.UUID(int=10)))
    with pytest.raises(RuntimeError):
        campaign.validate_snapshot(candidate, require_namespace=True)
    accepted = json.loads(json.dumps(baseline))
    set_member_identity(accepted, node, 11, str(uuid.UUID(int=11)))
    campaign.validate_snapshot(accepted, require_namespace=True)
    reappeared = json.loads(json.dumps(accepted))
    set_member_identity(reappeared, node, 12, str(uuid.UUID(int=10)))
    with pytest.raises(RuntimeError, match="retired node incarnation reappeared"):
        campaign.validate_snapshot(reappeared, require_namespace=True)


def test_quorum_phase_binds_prefault_absence_readiness_and_current_identity(
    tmp_path: Path,
):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    campaign.expected_orbitkv_endpoints["metadata-pressure"] = "127.0.0.1:26002"
    baseline = valid_snapshot(campaign)
    for index, node in enumerate(("a100-consumer", "metadata-pressure", "query-source"), start=1):
        set_member_identity(baseline, node, 10, str(uuid.UUID(int=100 + index)))
    campaign.validate_snapshot(baseline, require_namespace=True)
    prefault = {
        node: member_evidence_from_snapshot(campaign, baseline, node)
        for node in ("a100-consumer", "metadata-pressure", "query-source")
    }
    pressure_owner = prefault["metadata-pressure"]["record"]["member"]["owner"]
    campaign.validate_phase_evidence(
        "orbitkv_consumer",
        {
            "converged": {"metadata": {"index": {"registration_valid": True}}},
            "oracle": baseline,
            "members": prefault,
            "pressure_ready": {"owner": pressure_owner},
        },
    )

    recovered = json.loads(json.dumps(baseline))
    for index, node in enumerate(("a100-consumer", "metadata-pressure", "query-source"), start=1):
        set_member_identity(recovered, node, 11, str(uuid.UUID(int=200 + index)))
    recovered["namespace"]["header"]["revision"] = "30"
    for row in recovered["namespace"]["kvs"]:
        row["mod_revision"] = "22"
    for status in recovered["statuses"].values():
        status["header"]["revision"] = "30"
    campaign.validate_snapshot(recovered, require_namespace=True)
    current = {
        node: member_evidence_from_snapshot(campaign, recovered, node)
        for node in ("a100-consumer", "metadata-pressure", "query-source")
    }
    current_pressure = current["metadata-pressure"]["record"]["member"]
    installed = {
        "owner": current_pressure["owner"]["incarnation"],
        "fresh": True,
    }
    recovery = {
        "old_members": prefault,
        "old_member_absence": {
            node: member_absence_for_campaign(campaign, node, evidence, 25)
            for node, evidence in prefault.items()
        },
        "members": current,
        "pressure": {"ready": {"owner": current_pressure["owner"]}},
    }
    evidence = {
        "recovered": {
            "metadata": {"index": {"registration_valid": True}},
            "owners": [installed],
        },
        "recovery": recovery,
        "snapshot": recovered,
    }
    campaign.validate_phase_evidence("quorum_restore", evidence)

    forged_old = json.loads(json.dumps(recovery))
    forged_old["old_members"]["a100-consumer"]["record"]["member"]["epoch"] = 9
    with pytest.raises(RuntimeError, match="authenticated pre-fault"):
        campaign.validate_phase_evidence("quorum_restore", {**evidence, "recovery": forged_old})

    wrong_cluster_absence = json.loads(json.dumps(recovery))
    wrong_cluster_absence["old_member_absence"]["query-source"]["responses"][
        campaign.settings["nodes"][0]["client_url"]
    ]["header"] = {"cluster_id": "999", "revision": 0}
    with pytest.raises(RuntimeError, match="authenticated member absence"):
        campaign.validate_phase_evidence(
            "quorum_restore", {**evidence, "recovery": wrong_cluster_absence}
        )

    wrong_ready = json.loads(json.dumps(recovery))
    wrong_ready["pressure"]["ready"]["owner"]["incarnation"] = str(uuid.UUID(int=999))
    with pytest.raises(RuntimeError, match="pressure identity is not bound"):
        campaign.validate_phase_evidence("quorum_restore", {**evidence, "recovery": wrong_ready})


def test_snapshot_rejects_changed_persistent_format_uuid(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    baseline = valid_snapshot(campaign)
    campaign.validate_snapshot(baseline, require_namespace=True)
    changed = json.loads(json.dumps(baseline))
    row = changed["namespace"]["kvs"][0]
    value = json.loads(base64.b64decode(row["value"]))
    value["cluster_uuid"] = str(uuid.uuid4())
    row["value"] = base64.b64encode(json.dumps(value).encode()).decode()
    with pytest.raises(RuntimeError, match="persistent cluster format changed"):
        campaign.validate_snapshot(changed, require_namespace=True)


def test_revoked_query_source_member_cannot_remain(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    snapshot = valid_snapshot(campaign)
    campaign.validate_snapshot(snapshot, require_namespace=True)
    campaign.required_orbitkv_members.remove("query-source")
    with pytest.raises(RuntimeError, match="unexpected OrbitKV namespace keys"):
        campaign.validate_snapshot(snapshot, require_namespace=True)


@pytest.mark.parametrize(
    ("node", "expected", "wrong"),
    [
        ("metadata-pressure", "127.0.0.1:26002", "127.0.0.1:26007"),
        ("cold-observer", "127.0.0.1:26003", "127.0.0.1:26008"),
    ],
)
def test_snapshot_binds_dynamic_and_cold_member_endpoints(
    tmp_path: Path, node: str, expected: str, wrong: str
):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    snapshot = valid_snapshot(campaign)
    prefix = f"/orbitkv/v2/{campaign.settings['orbitkv_cluster']}/"
    if node == "cold-observer":
        campaign.required_orbitkv_members.add(node)
        snapshot["namespace"]["kvs"].extend(
            [
                kv(f"{prefix}epochs/{node}", "4", 0),
                kv(
                    f"{prefix}members/{node}",
                    {
                        "node_id": node,
                        "epoch": 4,
                        "owner": {
                            "endpoint": wrong,
                            "incarnation": str(uuid.uuid4()),
                        },
                        "protocol": campaign.settings["inventory_protocol"],
                    },
                    104,
                ),
            ]
        )
    else:
        row = next(
            row
            for row in snapshot["namespace"]["kvs"]
            if base64.b64decode(row["key"]).decode().endswith(f"members/{node}")
        )
        member = json.loads(base64.b64decode(row["value"]))
        member["owner"]["endpoint"] = wrong
        row["value"] = base64.b64encode(json.dumps(member).encode()).decode()
    campaign.expected_orbitkv_endpoints[node] = expected
    with pytest.raises(RuntimeError, match="runtime endpoint differs"):
        campaign.validate_snapshot(snapshot, require_namespace=True)


def test_cold_phase_binds_authenticated_member_to_live_runtime(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    campaign.required_orbitkv_members.add("cold-observer")
    snapshot = valid_snapshot(campaign)
    prefix = f"/orbitkv/v2/{campaign.settings['orbitkv_cluster']}/"
    cold_member = {
        "node_id": "cold-observer",
        "epoch": 4,
        "owner": {
            "endpoint": "127.0.0.1:26003",
            "incarnation": str(uuid.UUID(int=404)),
        },
        "protocol": campaign.settings["inventory_protocol"],
    }
    snapshot["namespace"]["kvs"].extend(
        [
            kv(f"{prefix}epochs/cold-observer", "4", 0),
            kv(f"{prefix}members/cold-observer", cold_member, 104),
        ]
    )
    campaign.validate_snapshot(snapshot, require_namespace=True)
    member_evidence = member_evidence_from_snapshot(campaign, snapshot, "cold-observer")
    preabsence = member_absence_for_campaign(campaign, "cold-observer", None, 20)
    absence = member_absence_for_campaign(campaign, "cold-observer", member_evidence, 21)
    cleanup = {
        "consumer": {
            "manager_exit": 0,
            "pressure": {"status": "stopped", "returncode": 0},
            "manager_cleanup": {"status": "already_released"},
            "query_source": {
                "status": "stopped",
                "termination": {"status": "stopped"},
                "manager_cleanup": {"status": "already_released"},
                "drain": {"orbitkv_transfer_lock_active": 0},
            },
            "manager_drain": {
                "orbitkv_query_reserved_bytes": 0,
                "orbitkv_query_reserved_bytes_by_phase": 0,
                "orbitkv_transfer_completion_outstanding": 0,
            },
        },
        "nodes": {name: {"status": "stopped", "returncode": 0} for name in campaign.nodes},
    }
    evidence = {
        "keys": [f"{prefix}format"],
        "final": snapshot,
        "storage": {"all_nodes": True},
        "cold": {
            "metadata": {"index": {"registration_valid": True}},
            "member_evidence": member_evidence,
            "controller_snapshot": snapshot,
            "preexisting_member_absence": preabsence,
            "member_absence": absence,
            "runtime_owner": {
                "node": "cold-observer",
                "status": "running",
                "pid": 123,
                "start_ticks": "456",
                "peer_endpoint": "127.0.0.1:26003",
                "process_live": True,
            },
            "controller_runtime_identity": {
                "pid": 123,
                "start_ticks": "456",
                "listener": ["127.0.0.1", 26003, 123],
                "raw": [],
            },
            "termination": {
                "status": "stopped",
                "returncode": 0,
                "sigterm_requested": True,
            },
            "manager_cleanup": {
                "status": "already_released",
                "returncode": 0,
            },
        },
        "cleanup": cleanup,
        "postflight": {"compute": []},
    }
    campaign.validate_phase_evidence("cold_observer_final_oracle", evidence)

    escaped = json.loads(json.dumps(evidence))
    escaped["cold"]["member_evidence"]["header"]["cluster_id"] = "999"
    with pytest.raises(RuntimeError, match="agent/controller member evidence differs"):
        campaign.validate_phase_evidence("cold_observer_final_oracle", escaped)

    unowned = json.loads(json.dumps(evidence))
    unowned["cold"]["runtime_owner"]["process_live"] = False
    with pytest.raises(RuntimeError, match="member/runtime identity differs"):
        campaign.validate_phase_evidence("cold_observer_final_oracle", unowned)


def test_snapshot_and_owner_watermarks_never_regress(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    campaign.last_write_revision = 21
    with pytest.raises(RuntimeError, match="revision regressed"):
        campaign.validate_snapshot(valid_snapshot(campaign), require_namespace=True)
    campaign.validate_owner_watermarks({"owners": [{"owner": "source", "applied_sequence": 100}]})
    with pytest.raises(RuntimeError, match="watermark regressed"):
        campaign.validate_owner_watermarks({"owners": [{"owner": "source", "applied_sequence": 0}]})


def test_phase_requires_real_old_source_query_rejection(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    owner = {
        "endpoint": "127.0.0.1:26002",
        "incarnation": str(uuid.UUID(int=42)),
    }
    new_member = {"node_id": "metadata-pressure", "epoch": 2, "owner": owner}
    campaign.validate_agent_member_evidence = lambda *_args: new_member
    campaign.owner = lambda _snapshot, _incarnation: {"fresh": True}
    evidence = {
        "new_member": new_member,
        "revoke": {"old_member": {"epoch": 1}},
        "replacement_result": {"final_records": ["record"]},
        "replacement_installed": {"pressure_poll": None, "owners": []},
        "replacement_snapshot": {},
        "replacement": {
            "member": {"authenticated": True},
            "replacement": {"ready": {"owner": owner}},
        },
        "query_source_member": {"owner": {"endpoint": "127.0.0.1:26005"}},
        "query_source_revoke": {"member_absent": True},
        "stale_query_preparation": {"local_resaved": True},
        "live_source_query": {
            "hash": "00",
            "source_endpoint": "127.0.0.1:26005",
            "expected_hits": 1,
            "observed_hits": 1,
            "lease_observed": True,
            "query_reserved_bytes_after": 0,
        },
        "stale_source_query": {
            "hash": "00",
            "source_endpoint": "127.0.0.1:26005",
            "expected_hits": 0,
            "observed_hits": 1,
            "lease_observed": False,
            "query_reserved_bytes_after": 0,
        },
    }
    evidence["replacement"]["replacement"]["ready"]["owner"] = {
        **owner,
        "incarnation": str(uuid.UUID(int=43)),
    }
    with pytest.raises(RuntimeError, match="replacement pressure identity is not bound"):
        campaign.validate_phase_evidence("lease_revoke_replacement", evidence)
    evidence["replacement"]["replacement"]["ready"]["owner"] = owner
    with pytest.raises(RuntimeError, match="live/stale source query control differs"):
        campaign.validate_phase_evidence("lease_revoke_replacement", evidence)
    evidence["stale_source_query"]["observed_hits"] = 0
    campaign.validate_phase_evidence("lease_revoke_replacement", evidence)


def test_external_owner_uses_current_status_pid_identity(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    owner = {
        "node": "s2-a100",
        "status_path": "/tmp/current-status.json",
        "required": True,
        "terminal_states": ["COMPLETE"],
    }
    calls = []

    def command(node, argv, **kwargs):
        del node, kwargs
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0 if argv[0] == "test" else 1, "", "")

    campaign.host_command = command
    campaign.remote_json = lambda *_args, **_kwargs: {
        "state": "COMPLETE",
        "controller_pid": 1234,
        "controller_start_ticks": "current-ticks",
    }
    with pytest.raises(RuntimeError, match="still present"):
        campaign.external_owner_evidence(owner)
    assert any("1234" in argv and "current-ticks" in argv for argv in calls)


def wait_path(path: Path, process: subprocess.Popen, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"process exited before {path}: {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(path)
        time.sleep(0.02)


def free_ipv6_port() -> int:
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
        listener.bind(("::1", 0))
        return listener.getsockname()[1]


def test_listener_evidence_requires_exact_address_port_and_child_pid(tmp_path: Path):
    root, config, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    campaign.output.mkdir()
    node = config["nodes"][0]
    child_record = {"pid": 4242, "start_ticks": "current", "run_index": 0}
    campaign.remote_json = lambda *_args, **_kwargs: {
        "state": "running",
        "child": child_record,
    }

    def command(_node, argv, **_kwargs):
        if argv[0] == "python3":
            return subprocess.CompletedProcess(argv, 0, "", "")
        wrong = (
            f"LISTEN 0 128 [{node['ipv6']}]:{config['etcd']['client_port']} "
            f'[::]:* users:(("etcd",pid=9999,fd=3))\n'
            f"LISTEN 0 128 [{node['ipv6']}]:{config['etcd']['peer_port']} "
            f'[::]:* users:(("etcd",pid=9999,fd=4))\n'
        )
        return subprocess.CompletedProcess(argv, 0, wrong, "")

    campaign.host_command = command
    with pytest.raises(RuntimeError, match="exact listener identity differs"):
        campaign.listener_evidence([node["name"]])

    def correct(_node, argv, **_kwargs):
        if argv[0] == "python3":
            return subprocess.CompletedProcess(argv, 0, "", "")
        output = (
            f"LISTEN 0 128 [{node['ipv6']}]:{config['etcd']['client_port']} "
            f'[::]:* users:(("etcd",pid=4242,fd=3))\n'
            f"LISTEN 0 128 [{node['ipv6']}]:{config['etcd']['peer_port']} "
            f'[::]:* users:(("etcd",pid=4242,fd=4))\n'
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    campaign.host_command = correct
    evidence = campaign.listener_evidence([node["name"]])
    assert evidence[node["name"]]["child"] == child_record


def test_listener_identity_ignores_unrelated_wildcard_listener():
    line = 'LISTEN 0 4096 *:1234 *:* users:(("other",pid=42,fd=7))'
    assert campaign_module.Campaign.listener_identity(line) is None
    assert campaign_module.Campaign.listener_port(line) == 1234


def test_port_gate_rejects_wildcard_listener_without_binding(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    commands = []

    def command(_node, argv, **_kwargs):
        commands.append(argv)
        output = 'LISTEN 0 4096 *:24000 *:* users:(("other",pid=42,fd=7))\n'
        return subprocess.CompletedProcess(argv, 0, output, "")

    campaign.host_command = command
    with pytest.raises(RuntimeError, match="preflight has live listeners"):
        campaign.require_ports_without_listeners(
            campaign.nodes["s2-h20"],
            {24000, 25000},
            context="unit preflight",
        )
    assert commands == [["ss", "-H", "-ltnp"]]


def test_port_gate_accepts_time_wait_without_listener(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)

    def command(_node, argv, **_kwargs):
        assert argv == ["ss", "-H", "-ltnp"]
        return subprocess.CompletedProcess(argv, 0, "", "")

    campaign.host_command = command
    assert campaign.require_ports_without_listeners(
        campaign.nodes["s2-h20"],
        {24000, 25000},
        context="unit postflight",
        cleanup=True,
    ) == {"status": "no_listener", "ports": [24000, 25000]}


def test_consumer_launch_waits_for_running_state(tmp_path: Path, monkeypatch):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    campaign.output.mkdir()
    launch = {"pid": 42, "start_ticks": "99", "status": "launched"}
    states = iter([{"state": "starting"}, {"state": "running", "manager_pid": 7}])
    campaign.host_command = lambda *_args, **_kwargs: subprocess.CompletedProcess(
        [], 0, json.dumps(launch), ""
    )
    campaign.wait_remote_json = lambda *_args, **_kwargs: next(states)
    campaign.remote_json = lambda *_args, **_kwargs: next(states)
    monkeypatch.setattr(campaign_module.time, "sleep", lambda _seconds: None)

    assert campaign.launch_consumer()["state"] == "running"
    assert campaign.consumer_launched
    assert campaign.state["consumer"] == launch


def test_consumer_terminal_start_failure_releases_confirmed_owner(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    campaign.output.mkdir()
    launch = {"pid": 42, "start_ticks": "99", "status": "launched"}
    terminal = {
        "state": "invalid",
        "failure_cleanup": {
            "quarantined": False,
            "uds_root_cleanup": {"status": "released"},
        },
    }
    campaign.host_command = lambda *_args, **_kwargs: subprocess.CompletedProcess(
        [], 0, json.dumps(launch), ""
    )
    campaign.wait_remote_json = lambda *_args, **_kwargs: terminal
    observed = []
    campaign.wait_process_absent = lambda node, record, **_kwargs: observed.append(
        (node["name"], record)
    )

    with pytest.raises(RuntimeError, match="consumer did not start"):
        campaign.launch_consumer()
    assert observed == [("s2-a100", launch)]
    assert not campaign.consumer_launched
    assert campaign.state["consumer"]["status"] == "startup_failed_reaped"


def test_readiness_cleanup_stops_consumer_before_etcd_nodes():
    events = []
    campaign = types.SimpleNamespace(
        settings={"budgets": {"cleanup_seconds": 180}},
        cleanup_deadline=None,
        network_probes={},
        consumer_launched=True,
        launched_nodes={"s2-h20"},
        consumer_request=lambda *_args, **_kwargs: events.append("consumer") or {"manager_exit": 0},
        node_request=lambda name, *_args, **_kwargs: (
            events.append(name) or {"status": "terminated_by_sigterm"}
        ),
        postflight=lambda: events.append("postflight") or {"status": "passed"},
        release_resource_locks=lambda: events.append("locks") or {"s2-h20": {"status": "released"}},
    )

    cleanup = readiness_module._cleanup(campaign)
    assert events == ["consumer", "s2-h20", "postflight", "locks"]
    assert "errors" not in cleanup


def test_cold_manager_listener_binds_owned_pid_and_frozen_port(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign = campaign_module.Campaign(root, digest)
    runtime_owner = {
        "pid": 4242,
        "start_ticks": "current",
    }

    def command(_node, argv, **_kwargs):
        if argv[0] == "python3":
            return subprocess.CompletedProcess(argv, 0, "", "")
        output = (
            "LISTEN 0 128 127.0.0.1:26003 0.0.0.0:* "
            'users:(("orbitkv-cache-manager",pid=9999,fd=3))\n'
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    campaign.host_command = command
    with pytest.raises(RuntimeError, match="cold Manager exact listener identity differs"):
        campaign.manager_listener_evidence(runtime_owner)

    def correct(_node, argv, **_kwargs):
        if argv[0] == "python3":
            return subprocess.CompletedProcess(argv, 0, "", "")
        output = (
            "LISTEN 0 128 127.0.0.1:26003 0.0.0.0:* "
            'users:(("orbitkv-cache-manager",pid=4242,fd=3))\n'
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    campaign.host_command = correct
    evidence = campaign.manager_listener_evidence(runtime_owner)
    assert evidence["listener"] == ["127.0.0.1", 26003, 4242]

    def consumer_listener(_node, argv, **_kwargs):
        if argv[0] == "python3":
            return subprocess.CompletedProcess(argv, 0, "", "")
        output = (
            "LISTEN 0 128 127.0.0.1:26001 0.0.0.0:* "
            'users:(("orbitkv-cache-manager",pid=4242,fd=3))\n'
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    campaign.host_command = consumer_listener
    evidence = campaign.manager_listener_evidence(
        runtime_owner, peer_port=26001, label="readiness Manager"
    )
    assert evidence["listener"] == ["127.0.0.1", 26001, 4242]


def test_network_probe_binds_exact_ports_and_records_source(tmp_path: Path):
    root, config, _ = staged_root(tmp_path)
    ports = [free_ipv6_port(), free_ipv6_port()]
    while ports[1] == ports[0]:
        ports[1] = free_ipv6_port()
    config["etcd"]["client_port"], config["etcd"]["peer_port"] = ports
    for node in config["nodes"]:
        node["ipv6"] = "::1"
    (root / "inputs/settings.json").write_text(json.dumps(config, indent=2) + "\n")
    digest = write_manifest(root)
    process = subprocess.Popen(
        [
            sys.executable,
            str(root / "inputs/network_probe_agent.py"),
            "--root",
            str(root),
            "--node",
            "s2-h20",
            "--approved-input-manifest-sha256",
            digest,
        ],
        env=direct_network_environment(config["network_environment"]),
        start_new_session=True,
    )
    try:
        wait_path(root / "s2-h20.network-probe.json", process)
        for port in ports:
            with socket.create_connection(("::1", port), timeout=2) as connection:
                connection.sendall(bytes([165]))
        time.sleep(0.1)
        (root / "s2-h20.network-probe.stop").touch()
        assert process.wait(timeout=3) == 0
        result = json.loads((root / "s2-h20.network-probe.result.json").read_text())
        assert result["status"] == "stopped"
        assert {(row["port"], row["payload_hex"]) for row in result["connections"]} == {
            (port, "a5") for port in ports
        }
        assert {row["source_ipv6"] for row in result["connections"]} == {"::1"}
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


def test_resource_lock_is_real_and_exclusive(tmp_path: Path):
    first, first_config, _ = staged_root(tmp_path, "first")
    second, second_config, _ = staged_root(tmp_path, "second")
    for root, config in ((first, first_config), (second, second_config)):
        config["budgets"].update(
            {"global_seconds": 0.01, "cleanup_seconds": 0.01, "command_seconds": 0.01}
        )
        (root / "inputs/settings.json").write_text(json.dumps(config, indent=2) + "\n")
    first_digest = write_manifest(first)
    second_digest = write_manifest(second)
    script = first / "inputs/resource_lock_agent.py"
    first_process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--root",
            str(first),
            "--node",
            "s2-h20",
            "--approved-input-manifest-sha256",
            first_digest,
        ],
        env=direct_network_environment(first_config["network_environment"]),
        start_new_session=True,
    )
    try:
        wait_path(first / "s2-h20.resource-lock.json", first_process)
        time.sleep(0.1)
        assert first_process.poll() is None
        second_process = subprocess.run(
            [
                sys.executable,
                str(second / "inputs/resource_lock_agent.py"),
                "--root",
                str(second),
                "--node",
                "s2-h20",
                "--approved-input-manifest-sha256",
                second_digest,
            ],
            env=direct_network_environment(second_config["network_environment"]),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        assert second_process.returncode != 0
        (first / "s2-h20.resource-lock.stop").touch()
        assert first_process.wait(timeout=3) == 0
        released = json.loads((first / "s2-h20.resource-lock.released.json").read_text())
        assert released["status"] == "released"
    finally:
        if first_process.poll() is None:
            os.killpg(first_process.pid, signal.SIGKILL)
            first_process.wait(timeout=2)


def test_input_manifest_digest_and_helper_coverage_are_mandatory(tmp_path: Path):
    root, _, digest = staged_root(tmp_path)
    campaign_module.Campaign(root, digest)
    with pytest.raises(RuntimeError, match="approved digest"):
        campaign_module.Campaign(root, "0" * 64)
    manifest = root / "inputs/SHA256SUMS"
    manifest.write_text(
        "".join(
            line
            for line in manifest.read_text().splitlines(keepends=True)
            if not line.endswith("  consumer_agent.py\n")
        )
    )
    reduced = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="omits imported helpers"):
        campaign_module.Campaign(root, reduced)


def test_production_harness_has_no_implicit_subprocess_timeout_kill():
    source = Path(consumer_module.__file__).read_text()
    assert "subprocess.run(" not in source
    lock_source = (Path(campaign_module.__file__).parent / "resource_lock_agent.py").read_text()
    assert '"expired"' not in lock_source


def test_frozen_configuration_guard_remains_active_under_optimized_python(tmp_path):
    root, _, digest = staged_root(tmp_path)
    settings_path = root / "inputs/settings.json"
    config = json.loads(settings_path.read_text())
    config["formal_launch_authorized"] = True
    settings_path.write_text(json.dumps(config) + "\n")
    digest = write_manifest(root)
    command = [
        sys.executable,
        "-O",
        "-m",
        "benches.three_host_etcd.validate_inputs",
        "--root",
        str(root),
        "--approved-input-manifest-sha256",
        digest,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode != 0
    assert "formal_launch_authorized" in result.stderr


def test_remote_cleanup_budget_never_renews_a_local_deadline(tmp_path, monkeypatch):
    agent = bare_agent(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(consumer_module.time, "monotonic", lambda: clock[0])
    agent.accept_cleanup_budget(0.8)
    assert agent.cleanup_remaining(10) == pytest.approx(0.8)
    clock[0] += 0.5
    agent.accept_cleanup_budget(0.8)
    assert agent.cleanup_remaining(10) == pytest.approx(0.3)
    clock[0] += 0.4
    with pytest.raises(TimeoutError, match="cumulative cleanup"):
        agent.cleanup_remaining(10)


@pytest.mark.parametrize("invalid", [True, -1, 0, float("nan"), float("inf"), "1", 181])
def test_remote_cleanup_rejects_invalid_budget(tmp_path, invalid):
    agent = bare_agent(tmp_path)
    with pytest.raises(RuntimeError, match="invalid remaining cleanup"):
        agent.accept_cleanup_budget(invalid)


def test_readiness_failure_cleanup_cannot_reset_an_existing_deadline(monkeypatch):
    campaign = types.SimpleNamespace(
        cleanup_deadline=100.0,
        settings={"budgets": {"cleanup_seconds": 180}},
        network_probes={},
        consumer_launched=False,
        launched_nodes=set(),
        postflight=lambda: {},
        release_resource_locks=lambda: {},
    )
    monkeypatch.setattr(readiness_module.time, "monotonic", lambda: 500.0)
    readiness_module._cleanup(campaign)
    assert campaign.cleanup_deadline == 100.0


def test_missing_lock_path_retains_live_owner_and_original_flock(tmp_path):
    root, config, digest = staged_root(tmp_path)
    process = subprocess.Popen(
        [
            sys.executable,
            str(root / "inputs/resource_lock_agent.py"),
            "--root",
            str(root),
            "--node",
            "s2-h20",
            "--approved-input-manifest-sha256",
            digest,
        ],
        env=direct_network_environment(config["network_environment"]),
        start_new_session=True,
    )
    try:
        ready = root / "s2-h20.resource-lock.json"
        wait_path(ready, process)
        lock_path = Path(config["resource_lock"]["path"])
        lock_path.unlink()
        deadline = time.monotonic() + 3
        while json.loads(ready.read_text())["status"] != "quarantined":
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert process.poll() is None
        assert lock_path.with_suffix(".quarantine.json").exists()
        assert not (root / "s2-h20.resource-lock.released.json").exists()
        (root / "s2-h20.resource-lock.stop").touch()
        time.sleep(0.05)
        assert process.poll() is None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def test_actual_import_origins_and_bytes_are_bound_to_payload(tmp_path, monkeypatch):
    agent = bare_agent(tmp_path)
    names = [
        "orbitkv",
        "orbitkv.client.gpu",
        "tests.integration.test_distributed_cache",
        "tests.support.cache_manager",
        "tests.support.metrics",
    ]
    origins, payload, modules = {}, {}, {}
    for index, name in enumerate(names):
        relative = f"runtime-venv/lib/python3.11/site-packages/module-{index}.py"
        path = agent.runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen helper\n")
        origins[name] = relative
        payload[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        modules[name] = types.SimpleNamespace(__file__=str(path))
    monkeypatch.setattr(consumer_module.importlib, "import_module", lambda name: modules[name])
    runtime = {
        "import_origins": origins,
        "installed_payload_roots": ["runtime-venv/lib/python3.11/site-packages"],
    }
    monkeypatch.setattr(
        consumer_module.sys, "pycache_prefix", str(agent.root / ".consumer-bytecode")
    )
    monkeypatch.setattr(
        consumer_module.sys,
        "flags",
        types.SimpleNamespace(dont_write_bytecode=True, ignore_environment=True, no_user_site=True),
    )
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(agent.root / ".consumer-bytecode"))
    assert len(agent.verify_imported_helpers(runtime, payload, {})) == 5
    changed = Path(modules["orbitkv.client.gpu"].__file__)
    changed.write_text("# changed installed code\n")
    with pytest.raises(RuntimeError, match="hash differs"):
        agent.verify_imported_helpers(runtime, payload, {})
    changed.write_text("# frozen helper\n")
    modules["tests.support.cache_manager"].__file__ = str(tmp_path / "shadow.py")
    (tmp_path / "shadow.py").write_text("# shadow helper\n")
    with pytest.raises(RuntimeError, match="origin differs"):
        agent.verify_imported_helpers(runtime, payload, {})
