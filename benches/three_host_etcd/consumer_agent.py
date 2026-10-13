"""Own the A100 OrbitKV consumer, local tensor and Rust inventory sources."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib
import ipaddress
import json
import math
import os
import re
import socket
import stat
import subprocess
import time
import uuid
from pathlib import Path

import requests

try:
    from .environment import (
        direct_network_environment,
        direct_network_environment_evidence,
        validated_manager_socket_paths,
    )
    from .integrity import verify_inputs, verify_manifest
except ImportError:
    from environment import (
        direct_network_environment,
        direct_network_environment_evidence,
        validated_manager_socket_paths,
    )
    from integrity import verify_inputs, verify_manifest
try:
    from tests.integration.test_distributed_cache import _query_ready
    from tests.support.cache_manager import CacheManagerProcess
    from tests.support.metrics import fetch_orbitkv_metrics
except ModuleNotFoundError:
    _query_ready = None
    CacheManagerProcess = None
    fetch_orbitkv_metrics = None

PRESSURE_TEST = "cluster::inventory::tests::pressure::external_metadata_only_pressure_source"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def start_ticks(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def listener_port(line: str) -> int | None:
    fields = line.split()
    if len(fields) < 5:
        return None
    try:
        return int(fields[3].rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


class Agent:
    def __init__(self, root: Path, settings: dict, approved_input_manifest_sha256: str):
        self.root = root
        self.settings = settings
        self.approved_input_manifest_sha256 = approved_input_manifest_sha256
        self.environment = direct_network_environment(settings["network_environment"])
        self.output = root / "consumer"
        self.controls = self.output / "controls"
        self.responses = self.output / "responses"
        self.runtime = Path(settings["orbitkv_runtime"]["source_root"])
        self.manager_socket_paths = validated_manager_socket_paths(settings["orbitkv_runtime"])
        self.uds_root = next(iter(self.manager_socket_paths.values())).parent
        self.uds_root_owner: dict | None = None
        self.etcd_session = requests.Session()
        self.etcd_session.trust_env = False
        self.etcd_cluster_id: str | None = None
        self.etcd_revisions: dict[str, int] = {}
        self.site = self.runtime / "runtime-venv/lib/python3.11/site-packages"
        self.manager: CacheManagerProcess | None = None
        self.query_source_manager: CacheManagerProcess | None = None
        self.cold_observer_manager: CacheManagerProcess | None = None
        self.cold_observer_member: dict | None = None
        self.manager_owners: list[CacheManagerProcess] = []
        self.failed_managers: list[CacheManagerProcess] = []
        self.failed_manager_cleanups: list[dict] = []
        self.client = None
        self.query_source_client = None
        self.tensor = None
        self.expected = None
        self.hashes = None
        self.query_source_tensor = None
        self.query_source_hashes = None
        self.pressure: subprocess.Popen | None = None
        self.pressure_log = None
        self.pressure_run = -1
        self.pressure_output: Path | None = None
        self.last_pressure_result: dict | None = None
        self.current_members: dict[str, dict] = {}
        self.cleanup_deadline: float | None = None
        self.manager_run = 0
        self.processed = []
        self.state = {
            "state": "starting",
            "agent_pid": os.getpid(),
            "agent_start_ticks": start_ticks(os.getpid()),
            "processed": self.processed,
            "manager_owners": [],
            "pressure_runs": [],
            "started_unix_ns": time.time_ns(),
            "network_environment": direct_network_environment_evidence(
                self.environment, settings["network_environment"]
            ),
            "uds_root": str(self.uds_root),
            "uds_root_owner": None,
        }

    def accept_cleanup_budget(self, seconds: object) -> None:
        if seconds is None:
            return
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
            or seconds > self.settings["budgets"]["cleanup_seconds"]
        ):
            raise RuntimeError("invalid remaining cleanup budget")
        deadline = time.monotonic() + seconds
        self.cleanup_deadline = min(self.cleanup_deadline or deadline, deadline)

    def cleanup_remaining(self, timeout: float) -> float:
        if self.cleanup_deadline is None:
            return timeout
        remaining = self.cleanup_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("consumer cumulative cleanup budget exhausted")
        return min(timeout, remaining)

    def save_state(self) -> None:
        self.state["updated_unix_ns"] = time.time_ns()
        atomic_json(self.output / "agent-status.json", self.state)

    def prepare_uds_root(self) -> dict:
        if self.uds_root.exists() or self.uds_root.is_symlink():
            raise FileExistsError(f"Manager UDS root already exists: {self.uds_root}")
        self.uds_root.mkdir(mode=0o700)
        metadata = self.uds_root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise RuntimeError(f"Manager UDS root ownership differs: {self.uds_root}")
        owner = {
            "path": str(self.uds_root),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "uid": metadata.st_uid,
            "mode": stat.S_IMODE(metadata.st_mode),
        }
        self.uds_root_owner = owner
        self.state["uds_root_owner"] = owner
        probes = {}
        for name, path in self.manager_socket_paths.items():
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Manager UDS path already exists: {path}")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.bind(str(path))
                socket_metadata = path.lstat()
                if not stat.S_ISSOCK(socket_metadata.st_mode):
                    raise RuntimeError(f"Manager UDS probe is not a socket: {path}")
            path.unlink()
            probes[name] = {
                "path": str(path),
                "path_bytes": len(os.fsencode(path)),
                "bind": "passed",
                "removed": True,
            }
        return {"owner": owner, "paths": probes}

    def release_uds_root(self) -> dict:
        owner = self.uds_root_owner
        if owner is None:
            return {"status": "not_created", "path": str(self.uds_root)}
        try:
            metadata = self.uds_root.lstat()
        except FileNotFoundError:
            return {"status": "ownership_lost", "owner": owner}
        identity = {
            "path": str(self.uds_root),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "uid": metadata.st_uid,
            "mode": stat.S_IMODE(metadata.st_mode),
        }
        if identity != owner or not stat.S_ISDIR(metadata.st_mode):
            return {"status": "ownership_changed", "owner": owner, "observed": identity}
        entries = sorted(path.name for path in self.uds_root.iterdir())
        if entries:
            return {"status": "retained_nonempty", "owner": owner, "entries": entries}
        self.uds_root.rmdir()
        self.uds_root_owner = None
        self.state["uds_root_owner"] = None
        return {"status": "released", "owner": owner}

    def metadata(self, manager: CacheManagerProcess | None = None) -> dict:
        selected = manager or self.manager
        response = self.etcd_session.get(
            f"http://127.0.0.1:{selected.http_port}/cache/metadata", timeout=5
        )
        response.raise_for_status()
        return response.json()

    def owners(self, manager: CacheManagerProcess | None = None) -> list[dict]:
        selected = manager or self.manager
        response = self.etcd_session.get(
            f"http://127.0.0.1:{selected.http_port}/cache/metadata/owners",
            params={"limit": 128},
            timeout=5,
        )
        response.raise_for_status()
        return response.json()

    def authenticate_etcd_header(self, endpoint: str, response: dict) -> dict:
        header = response.get("header")
        if not isinstance(header, dict):
            raise RuntimeError(f"etcd response header is absent: {response}")
        cluster_id = str(header.get("cluster_id", ""))
        revision = int(header.get("revision", 0))
        if not cluster_id or int(cluster_id) <= 0 or revision <= 0:
            raise RuntimeError(f"etcd response header is invalid: {header}")
        if self.etcd_cluster_id is None:
            self.etcd_cluster_id = cluster_id
        elif cluster_id != self.etcd_cluster_id:
            raise RuntimeError(
                f"etcd cluster identity changed: {cluster_id} != {self.etcd_cluster_id}"
            )
        previous_revision = self.etcd_revisions.get(endpoint, 0)
        if revision < previous_revision:
            raise RuntimeError(
                f"etcd revision regressed at {endpoint}: {revision} < {previous_revision}"
            )
        self.etcd_revisions[endpoint] = revision
        return {"cluster_id": cluster_id, "revision": revision}

    def decode_member_evidence(self, node: str, endpoint: str, response: dict) -> dict:
        key = f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/members/{node}"
        header = self.authenticate_etcd_header(endpoint, response)
        rows = response.get("kvs", [])
        if len(rows) != 1:
            raise RuntimeError(f"expected one {node} member: {rows}")
        row = rows[0]
        observed_key = base64.b64decode(row.get("key", "")).decode(errors="strict")
        lease = int(row.get("lease", 0))
        create_revision = int(row.get("create_revision", 0))
        mod_revision = int(row.get("mod_revision", 0))
        member = json.loads(base64.b64decode(row.get("value", "")))
        owner = member.get("owner", {})
        owner_endpoint = owner.get("endpoint", "")
        try:
            owner_host, owner_port = owner_endpoint.rsplit(":", 1)
            owner_address = ipaddress.ip_address(owner_host.strip("[]"))
            owner_port = int(owner_port)
        except (AttributeError, ValueError) as error:
            raise RuntimeError(f"invalid {node} member endpoint: {owner_endpoint}") from error
        if (
            observed_key != key
            or lease <= 0
            or not 0 < create_revision <= mod_revision <= header["revision"]
            or set(member) != {"node_id", "epoch", "owner", "protocol"}
            or member["node_id"] != node
            or int(member["epoch"]) <= 0
            or member["protocol"] != self.settings["inventory_protocol"]
            or set(owner) != {"endpoint", "incarnation"}
            or not owner_address.is_loopback
            or not 0 < owner_port <= 65535
            or uuid.UUID(owner["incarnation"]).int == 0
        ):
            raise RuntimeError(f"invalid {node} member evidence: {response}")
        return {
            "node": node,
            "requested_key": key,
            "endpoint": endpoint,
            "header": header,
            "record": {
                "key": observed_key,
                "lease": lease,
                "create_revision": create_revision,
                "mod_revision": mod_revision,
                "member": member,
            },
            "observed_unix_ns": time.time_ns(),
        }

    def etcd_member(self, node: str) -> dict:
        key = f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/members/{node}"
        payload = {"key": base64.b64encode(key.encode()).decode()}
        last_error = None
        for endpoint in (entry["client_url"] for entry in self.settings["nodes"]):
            try:
                response = self.etcd_session.post(
                    f"{endpoint}/v3/kv/range", json=payload, timeout=5
                )
                response.raise_for_status()
                body = response.json()
                if len(body.get("kvs", [])) != 1:
                    last_error = RuntimeError(f"expected one {node} member: {body.get('kvs', [])}")
                    continue
                return self.decode_member_evidence(node, endpoint, body)
            except requests.RequestException as error:
                last_error = error
        raise RuntimeError(f"could not read {node} member: {last_error}")

    def wait_member_absent(self, node: str, previous: dict | None) -> dict:
        key = f"/orbitkv/v2/{self.settings['orbitkv_cluster']}/members/{node}"
        payload = {"key": base64.b64encode(key.encode()).decode()}
        deadline = time.monotonic() + 30
        last_error = None
        while time.monotonic() < deadline:
            empty_responses = {}
            for endpoint in (entry["client_url"] for entry in self.settings["nodes"]):
                try:
                    response = self.etcd_session.post(
                        f"{endpoint}/v3/kv/range", json=payload, timeout=5
                    )
                    response.raise_for_status()
                    body = response.json()
                    header = self.authenticate_etcd_header(endpoint, body)
                    rows = body.get("kvs", [])
                    if not rows:
                        empty_responses[endpoint] = {"header": header, "records": 0}
                        continue
                    current = self.decode_member_evidence(node, endpoint, body)
                    if (
                        previous is not None
                        and current["record"]["member"]["owner"]["incarnation"]
                        != previous["record"]["member"]["owner"]["incarnation"]
                    ):
                        raise RuntimeError(
                            f"{node} was reused before old-member absence: {current}"
                        )
                    last_error = RuntimeError(f"old {node} member remains: {current}")
                    break
                except requests.RequestException as error:
                    last_error = error
                    break
            if len(empty_responses) == len(self.settings["nodes"]):
                return {
                    "node": node,
                    "requested_key": key,
                    "responses": empty_responses,
                    "previous_record": previous,
                    "member_absent": True,
                    "observed_unix_ns": time.time_ns(),
                }
            time.sleep(0.1)
        raise TimeoutError(f"old {node} member did not disappear: {previous}: {last_error}")

    def wait_etcd_member(self, node: str, expected_owner: dict | None = None) -> dict:
        deadline = time.monotonic() + 30
        last_error = None
        while time.monotonic() < deadline:
            try:
                evidence = self.etcd_member(node)
                member = evidence["record"]["member"]
                runtime = self.settings["orbitkv_runtime"]
                expected_endpoints = {
                    "a100-consumer": f"127.0.0.1:{runtime['manager_peer_port']}",
                    "query-source": f"127.0.0.1:{runtime['query_source_peer_port']}",
                    "cold-observer": f"127.0.0.1:{runtime['observer_peer_port']}",
                }
                if member["owner"]["endpoint"] != expected_endpoints.get(
                    node, member["owner"]["endpoint"]
                ) or (expected_owner is not None and member["owner"] != expected_owner):
                    raise RuntimeError(
                        f"{node} member/runtime owner differs: {member}, {expected_owner}"
                    )
                return evidence
            except (RuntimeError, ValueError) as error:
                last_error = error
                time.sleep(0.1)
        raise TimeoutError(f"{node} member did not appear: {last_error}")

    def capture_current_members(self, pressure_ready: dict) -> dict[str, dict]:
        pressure_owner = pressure_ready.get("owner")
        if not isinstance(pressure_owner, dict):
            raise RuntimeError(f"pressure readiness owner is absent: {pressure_ready}")
        return {
            "a100-consumer": self.wait_etcd_member("a100-consumer"),
            "metadata-pressure": self.wait_etcd_member(
                "metadata-pressure", expected_owner=pressure_owner
            ),
            "query-source": self.wait_etcd_member("query-source"),
        }

    def start_manager(
        self, *, observer: bool = False, query_source: bool = False
    ) -> CacheManagerProcess:
        runtime = self.settings["orbitkv_runtime"]
        if observer and query_source:
            raise ValueError("Manager role is ambiguous")
        if observer:
            peer_port = runtime["observer_peer_port"]
            http_port = runtime["observer_http_port"]
            node = "cold-observer"
        elif query_source:
            peer_port = runtime["query_source_peer_port"]
            http_port = runtime["query_source_http_port"]
            node = "query-source"
        else:
            peer_port = runtime["manager_peer_port"]
            http_port = runtime["manager_http_port"]
            node = "a100-consumer"
        run = self.manager_run
        self.manager_run += 1
        manager = CacheManagerProcess(
            peer_port,
            pool_size="64mb",
            http_port=http_port,
            bootstrap_socket=str(self.manager_socket_paths[node]),
            log_path=self.output / f"{node}-manager-{run:02d}.log",
            runtime_python_paths=(str(self.site),),
            extra_args=(
                "--etcd-endpoints",
                ",".join(node["client_url"] for node in self.settings["nodes"]),
                "--node-id",
                node,
                "--cluster-name",
                self.settings["orbitkv_cluster"],
                "--membership-ttl-secs",
                str(runtime["membership_ttl_seconds"]),
                "--index-budget",
                "16mb",
                "--inventory-journal-bytes",
                "16777216",
                "--inventory-stream-coalesce-ms",
                "2",
                "--metadata-namespace",
                runtime["metadata_namespace"],
            ),
        )
        self.manager_owners.append(manager)
        owner_record = {
            "node": node,
            "run": run,
            "status": "constructed",
            "pid": None,
            "start_ticks": None,
            "network_environment": direct_network_environment_evidence(
                self.environment, self.settings["network_environment"]
            ),
        }
        self.state["manager_owners"].append(owner_record)
        self.save_state()
        try:
            started = manager.start()
        except BaseException as error:
            cleanup = self._stop_manager_owner(manager, timeout=5)
            atomic_json(
                self.output / f"{node}-manager-{run:02d}-start-failure-cleanup.json",
                cleanup,
            )
            owner_record.update(
                {
                    "status": cleanup["status"],
                    "pid": cleanup.get("pid"),
                    "start_ticks": cleanup.get("start_ticks"),
                    "error": repr(error),
                }
            )
            self.save_state()
            raise RuntimeError(
                f"Manager start raised: cleanup={cleanup}; error={error!r}"
            ) from error
        if not started:
            cleanup = self._stop_manager_owner(manager, timeout=5)
            atomic_json(
                self.output / f"{node}-manager-{run:02d}-start-failure-cleanup.json",
                cleanup,
            )
            owner_record.update(
                {
                    "status": cleanup["status"],
                    "pid": cleanup.get("pid"),
                    "start_ticks": cleanup.get("start_ticks"),
                }
            )
            self.save_state()
            raise RuntimeError(f"Manager failed to start: cleanup={cleanup}\n{manager.read_logs()}")
        owner_record.update(
            {
                "status": "running",
                "pid": manager.process.pid,
                "start_ticks": start_ticks(manager.process.pid),
            }
        )
        self.save_state()
        return manager

    def _stop_manager_owner(self, manager: CacheManagerProcess, *, timeout: float) -> dict:
        try:
            cleanup = manager.stop_without_kill(timeout=self.cleanup_remaining(timeout))
        except BaseException as error:
            cleanup = {
                "status": "error",
                "error": repr(error),
                "pid": getattr(getattr(manager, "process", None), "pid", None),
            }
        safe = cleanup.get("status") in ("exited", "already_released") and cleanup.get(
            "returncode", 0
        ) in (None, 0)
        pid = cleanup.get("pid") or getattr(getattr(manager, "process", None), "pid", None)
        for record in self.state["manager_owners"]:
            if record.get("pid") == pid or (
                record.get("pid") is None and record.get("status") == "constructed"
            ):
                record.update(
                    {
                        "status": cleanup.get("status", "error"),
                        "cleanup": cleanup,
                    }
                )
                break
        if safe:
            if manager in self.manager_owners:
                self.manager_owners.remove(manager)
        elif manager not in self.failed_managers:
            self.failed_managers.append(manager)
            self.failed_manager_cleanups.append(cleanup)
        return cleanup

    def _terminate_manager_gracefully(
        self, manager: CacheManagerProcess, *, timeout: float
    ) -> dict:
        process = getattr(manager, "process", None)
        preexisting = process.poll() if process is not None else None
        if preexisting is not None:
            return {
                "status": "preexisting_exit",
                "returncode": preexisting,
                "sigterm_requested": False,
                "shutdown_seconds": 0.0,
            }
        code, seconds = manager.terminate_gracefully(timeout=self.cleanup_remaining(timeout))
        return {
            "status": "stopped" if code == 0 else "unexpected_exit",
            "returncode": code,
            "sigterm_requested": True,
            "shutdown_seconds": seconds,
        }

    def verify_imported_helpers(
        self, runtime: dict, payload_entries: dict, projection: dict
    ) -> dict:
        payload_roots = runtime.get("installed_payload_roots")
        if not isinstance(payload_roots, list) or not payload_roots:
            raise RuntimeError("complete installed payload roots are absent")
        observed_files = set()
        for root in payload_roots:
            if not isinstance(root, str) or Path(root).is_absolute() or ".." in Path(root).parts:
                raise RuntimeError("installed payload root escapes runtime")
            directory = (self.runtime / root).resolve(strict=True)
            if (
                not directory.is_relative_to(self.runtime.resolve(strict=True))
                or not directory.is_dir()
            ):
                raise RuntimeError("installed payload root is invalid")
            for path in directory.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    observed_files.add(path.relative_to(self.runtime).as_posix())
        if not observed_files.issubset(payload_entries):
            raise RuntimeError("installed payload manifest omits runtime files")
        required_imports = {
            "orbitkv",
            "orbitkv.client.gpu",
            "tests.integration.test_distributed_cache",
            "tests.support.cache_manager",
            "tests.support.metrics",
        }
        if not required_imports.issubset(runtime["import_origins"]):
            raise RuntimeError("frozen imported helper origins are incomplete")
        origins = {}
        for module_name, relative in runtime["import_origins"].items():
            module = importlib.import_module(module_name)
            actual = Path(module.__file__).resolve(strict=True)
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RuntimeError("frozen import origin escapes runtime")
            expected_path = (self.runtime / relative).resolve(strict=True)
            if (
                actual != expected_path
                or relative not in payload_entries
                and not relative.startswith("projection/")
            ):
                raise RuntimeError(
                    f"actual imported helper origin differs: {module_name}: {actual} != {expected_path}"
                )
            digest = hashlib.sha256(actual.read_bytes()).hexdigest()
            expected_digest = payload_entries.get(relative) or projection.get(
                relative.removeprefix("projection/")
            )
            if digest != expected_digest:
                raise RuntimeError(f"actual imported helper hash differs: {module_name}")
            origins[module_name] = {"path": str(actual), "sha256": digest}
        return origins

    def preflight(self) -> dict:
        import orbitkv
        import orbitkv.orbitkv as native

        if None in (_query_ready, CacheManagerProcess, fetch_orbitkv_metrics):
            raise RuntimeError("consumer runtime projection helpers are unavailable")
        verified_inputs = verify_inputs(self.root, self.approved_input_manifest_sha256)

        node = self.settings["nodes"][1]
        if socket.gethostname() != node["hostname"]:
            raise RuntimeError("consumer hostname mismatch")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if boot_id != node["boot_id"]:
            raise RuntimeError("consumer boot ID mismatch")
        runtime = self.settings["orbitkv_runtime"]
        projection_manifest = self.runtime / "projection/SHA256SUMS"
        projection = verify_manifest(
            self.runtime / "projection",
            projection_manifest,
            runtime["projection_manifest_sha256"],
        )
        payload_entries = verify_manifest(
            self.runtime,
            self.runtime / "frozen/installed-payload.SHA256SUMS",
            runtime["installed_payload_manifest_sha256"],
        )
        origins = self.verify_imported_helpers(runtime, payload_entries, projection)
        projection_digest = runtime["projection_manifest_sha256"]
        artifacts = {
            "manager": self.runtime / "frozen/bin/orbitkv-cache-manager",
            "server_tests": self.runtime / "frozen/bin/server-tests",
            "extension": Path(native.__file__).resolve(),
            "tent": self.runtime / "frozen/native/libtent_shared.so",
        }
        expected = {
            "manager": runtime["manager_sha256"],
            "server_tests": runtime["server_tests_sha256"],
            "extension": runtime["extension_sha256"],
            "tent": runtime["tent_sha256"],
        }
        observed = {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in artifacts.items()
        }
        if observed != expected:
            raise RuntimeError(f"consumer runtime hash mismatch: {observed} != {expected}")
        if not Path(orbitkv.__file__).resolve().is_relative_to(self.site):
            raise RuntimeError("consumer imported a source OrbitKV package")
        ports = {
            runtime["manager_peer_port"],
            runtime["manager_http_port"],
            runtime["observer_peer_port"],
            runtime["observer_http_port"],
            runtime["query_source_peer_port"],
            runtime["query_source_http_port"],
        }
        try:
            listeners = subprocess.check_output(
                ["ss", "-H", "-ltnp"],
                env=self.environment,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Manager listener inspection failed: {error.stderr.strip()}"
            ) from error
        conflicts = [line for line in listeners.splitlines() if listener_port(line) in ports]
        if conflicts:
            raise RuntimeError(f"Manager ports already have live listeners: {conflicts}")
        uds = self.prepare_uds_root()
        return {
            "input_manifest_sha256": self.approved_input_manifest_sha256,
            "input_files": sorted(verified_inputs),
            "projection_manifest_sha256": projection_digest,
            "projection_files": sorted(projection),
            "installed_payload_files": len(payload_entries),
            "import_origins": origins,
            "hostname": node["hostname"],
            "boot_id": boot_id,
            "package": str(Path(orbitkv.__file__).resolve()),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "hashes": observed,
            "uds": uds,
        }

    def start_local_cache(self) -> dict:
        import torch

        from orbitkv import CacheManagerClient
        from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

        runtime = self.settings["orbitkv_runtime"]
        self.manager = self.start_manager()
        self.client = CacheManagerClient(self.manager.bootstrap_socket)
        identity = runtime["local_identity"]
        pages, block_bytes = 8, 4096
        self.tensor = torch.empty(pages * block_bytes, dtype=torch.uint8, device="cuda")
        self.expected = torch.arange(pages * block_bytes, dtype=torch.int64, device="cuda").to(
            torch.uint8
        )
        self.tensor.copy_(self.expected)
        torch.cuda.synchronize()
        self.hashes = [
            hashlib.sha256(f"s2-etcd-local:{block}".encode()).digest() for block in range(pages)
        ]
        self.client.start_session_watcher("local", identity, 1, 1)
        ok, message = self.client.register_context_batch(
            "local",
            identity,
            0,
            0,
            1,
            1,
            resolve_device_id(),
            ["kv:0"],
            [serialize_gpu_buffer(self.tensor)],
            [pages],
            [block_bytes],
            [0],
            [1],
            "direct",
            False,
            tensors=[self.tensor],
        )
        if not ok:
            raise RuntimeError(message)
        ok, message = self.client.save(
            "local",
            0,
            0,
            resolve_device_id(),
            [("kv:0", list(range(pages)), self.hashes)],
        )
        if not ok:
            raise RuntimeError(message)
        deadline = time.monotonic() + 10
        expected_namespace = runtime["metadata_namespace"]
        while f"storage_namespace={expected_namespace}" not in self.manager.read_logs():
            if time.monotonic() >= deadline:
                raise RuntimeError("local identity did not resolve to the frozen namespace")
            time.sleep(0.05)
        check = self.check_local("initial")
        return {
            "identity": identity,
            "namespace": expected_namespace,
            "payload_sha256": check["payload_sha256"],
            "keys_sha256": hashlib.sha256(b"".join(self.hashes)).hexdigest(),
        }

    def start_query_source(self) -> dict:
        import torch

        from orbitkv import CacheManagerClient
        from orbitkv.client.gpu import resolve_device_id, serialize_gpu_buffer

        runtime = self.settings["orbitkv_runtime"]
        self.query_source_manager = self.start_manager(query_source=True)
        self.query_source_client = CacheManagerClient(self.query_source_manager.bootstrap_socket)
        identity = runtime["local_identity"]
        pages, block_bytes = 8, 4096
        self.query_source_tensor = (
            torch.arange(pages * block_bytes, dtype=torch.int64, device="cuda")
            .add(73)
            .remainder(251)
            .to(torch.uint8)
        )
        torch.cuda.synchronize()
        self.query_source_hashes = [
            hashlib.sha256(f"s2-etcd-query-source:{block}".encode()).digest()
            for block in range(pages)
        ]
        client = self.query_source_client
        client.start_session_watcher("query-source", identity, 1, 1)
        ok, message = client.register_context_batch(
            "query-source",
            identity,
            0,
            0,
            1,
            1,
            resolve_device_id(),
            ["kv:0"],
            [serialize_gpu_buffer(self.query_source_tensor)],
            [pages],
            [block_bytes],
            [0],
            [1],
            "direct",
            False,
            tensors=[self.query_source_tensor],
        )
        if not ok:
            raise RuntimeError(message)
        ok, message = client.save(
            "query-source",
            0,
            0,
            resolve_device_id(),
            [("kv:0", list(range(pages)), self.query_source_hashes)],
        )
        if not ok:
            raise RuntimeError(message)
        deadline = time.monotonic() + 10
        namespace = runtime["metadata_namespace"]
        while f"storage_namespace={namespace}" not in self.query_source_manager.read_logs():
            if time.monotonic() >= deadline:
                raise RuntimeError("query source did not resolve the frozen namespace")
            time.sleep(0.05)
        return {
            "identity": identity,
            "namespace": namespace,
            "hashes": [value.hex() for value in self.query_source_hashes],
            "payload_sha256": hashlib.sha256(
                self.query_source_tensor.cpu().numpy().tobytes()
            ).hexdigest(),
            "peer_endpoint": f"127.0.0.1:{runtime['query_source_peer_port']}",
        }

    def prepare_stale_source_query(self) -> dict:
        import torch

        from orbitkv.client.gpu import resolve_device_id

        response = self.etcd_session.post(
            f"http://127.0.0.1:{self.manager.http_port}/cache/memory/cleanup",
            timeout=30,
        )
        response.raise_for_status()
        cleanup = response.json()
        if cleanup["still_referenced_blocks"] != 0:
            raise RuntimeError(f"consumer cache cleanup retained blocks: {cleanup}")
        self.tensor.copy_(self.expected)
        torch.cuda.synchronize()
        ok, message = self.client.save(
            "local",
            0,
            0,
            resolve_device_id(),
            [("kv:0", list(range(len(self.hashes))), self.hashes)],
        )
        if not ok:
            raise RuntimeError(message)
        return {"cleanup": cleanup, "local_resaved": True}

    def check_local(self, phase: str) -> dict:
        import torch

        pages = len(self.hashes)
        query = _query_ready(
            self.client,
            "local",
            self.hashes,
            f"etcd-local-{phase}",
            pages,
            [self.manager],
        )
        self.tensor.zero_()
        torch.cuda.synchronize()
        operation = self.client.start_restore(
            "local",
            0,
            self.tensor.device.index or 0,
            [["kv:0"]],
            [(query.lease, [list(range(pages))])],
            ready_stream=torch.cuda.current_stream(self.tensor.device).cuda_stream,
        )
        status = self.client.wait_restore(operation, timeout=20)
        if not status.success:
            raise RuntimeError(f"local restore failed: {status}")
        torch.cuda.synchronize()
        if not torch.equal(self.tensor, self.expected):
            raise RuntimeError("local payload mismatch")
        return {
            "phase": phase,
            "payload_sha256": hashlib.sha256(self.tensor.cpu().numpy().tobytes()).hexdigest(),
            "metadata": self.metadata(),
            "observed_unix_ns": time.time_ns(),
        }

    def query_readable_source(self, *, expected_hits: int) -> dict:
        if not self.query_source_hashes:
            raise RuntimeError("readable query source is unavailable")
        block_hash = self.query_source_hashes[0]
        before = fetch_orbitkv_metrics(self.manager.http_port)
        if "orbitkv_query_reserved_bytes" not in before:
            raise RuntimeError("query reservation metric is absent before source query")
        query = _query_ready(
            self.client,
            "local",
            [block_hash],
            f"pressure-source-expected-{expected_hits}",
            expected_hits,
            [self.manager],
        )
        lease_observed = bool(query.lease)
        if expected_hits == 1:
            if not lease_observed:
                raise RuntimeError("live pressure source query lacks a lease")
            self.client.release(query.lease)
        elif expected_hits == 0:
            if lease_observed:
                raise RuntimeError("expired pressure source remained query-authorized")
        else:
            raise ValueError(f"unsupported expected hit count: {expected_hits}")
        deadline = time.monotonic() + 5
        while True:
            after = fetch_orbitkv_metrics(self.manager.http_port)
            if "orbitkv_query_reserved_bytes" not in after:
                raise RuntimeError("query reservation metric is absent after source query")
            if after["orbitkv_query_reserved_bytes"] == 0:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"source query lease did not drain: {after}")
            time.sleep(0.05)
        remote_metric = "orbitkv_remote_fetch_bytes_total"
        if remote_metric not in after:
            raise RuntimeError("remote fetch byte counter is absent after source query")
        remote_before = before.get(remote_metric, 0)
        remote_after = after[remote_metric]
        if expected_hits == 1 and remote_after < remote_before + 4096:
            raise RuntimeError(
                f"live source query did not fetch its real block: {remote_before} -> {remote_after}"
            )
        if expected_hits == 0 and (remote_metric not in before or remote_after != remote_before):
            raise RuntimeError(
                f"stale source query changed remote bytes: {remote_before} -> {remote_after}"
            )
        return {
            "hash": block_hash.hex(),
            "source_endpoint": (
                f"127.0.0.1:{self.settings['orbitkv_runtime']['query_source_peer_port']}"
            ),
            "expected_hits": expected_hits,
            "observed_hits": query.num_hit_blocks,
            "lease_observed": lease_observed,
            "query_reserved_bytes_before": before["orbitkv_query_reserved_bytes"],
            "query_reserved_bytes_after": after["orbitkv_query_reserved_bytes"],
            "remote_fetch_bytes_before": remote_before,
            "remote_fetch_bytes_after": remote_after,
            "observed_unix_ns": time.time_ns(),
        }

    def drain_metrics(self) -> dict:
        required = {
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
        }
        response = self.etcd_session.get(
            f"http://127.0.0.1:{self.manager.http_port}/metrics", timeout=self.cleanup_remaining(5)
        )
        response.raise_for_status()
        metrics = {}
        phase_labels = set()
        for line in response.text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            match = re.fullmatch(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)", line)
            name = line.split("{", 1)[0].split()[0]
            if name not in required:
                continue
            if match is None:
                raise RuntimeError(f"Malformed required drain metric: {line}")
            name, labels, raw_value = match.groups()
            try:
                value = float(raw_value)
            except ValueError as error:
                raise RuntimeError(f"Invalid required drain value: {line}") from error
            if not math.isfinite(value) or value != 0:
                raise RuntimeError(f"Manager drain metric is invalid or live: {name}={raw_value}")
            if name == "orbitkv_query_reserved_bytes_by_phase":
                phase_match = re.fullmatch(
                    r'phase="(warming|preloading|prepared|preparing|ready|restoring)"', labels or ""
                )
                if phase_match is None or phase_match[1] in phase_labels:
                    raise RuntimeError("Manager phase metric labels differ")
                phase_labels.add(phase_match[1])
            metrics[name] = metrics.get(name, 0) + value
        if phase_labels != {"warming", "preloading", "prepared", "preparing", "ready", "restoring"}:
            raise RuntimeError("Manager drain phase metrics are incomplete")
        missing = required - metrics.keys()
        if missing:
            raise RuntimeError(f"Manager drain metrics are absent: {sorted(missing)}")
        for name in sorted(required):
            value = metrics[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise RuntimeError(f"Manager drain metric is invalid: {name}={value}")
            if value != 0:
                raise RuntimeError(f"Manager did not drain {name}: {value}")
        return metrics

    def start_pressure(self, rounds: int) -> dict:
        if self.pressure is not None and self.pressure.poll() is None:
            raise RuntimeError("pressure source already running")
        runtime = self.settings["orbitkv_runtime"]
        self.pressure_run += 1
        artifact = self.output / f"pressure-{self.pressure_run:02d}"
        artifact.mkdir()
        go = artifact / "pressure.go"
        stop = artifact / "pressure.stop"
        environment = direct_network_environment(
            self.settings["network_environment"],
            source=self.environment,
            updates={
                "ORBITKV_PRESSURE_ETCD_ENDPOINTS": ",".join(
                    node["client_url"] for node in self.settings["nodes"]
                ),
                "ORBITKV_PRESSURE_CLUSTER": self.settings["orbitkv_cluster"],
                "ORBITKV_PRESSURE_NODE": "metadata-pressure",
                "ORBITKV_PRESSURE_NAMESPACE": runtime["metadata_namespace"],
                "ORBITKV_PRESSURE_ARTIFACT_DIR": str(artifact),
                "ORBITKV_PRESSURE_GO_FILE": str(go),
                "ORBITKV_PRESSURE_STOP_FILE": str(stop),
                "ORBITKV_PRESSURE_MODE": "pressure",
                "ORBITKV_PRESSURE_ROUNDS": str(rounds),
                "ORBITKV_PRESSURE_CADENCE_MS": str(runtime["pressure_cadence_ms"]),
                "ORBITKV_PRESSURE_KEYS": str(runtime["pressure_keys"]),
                "ORBITKV_PRESSURE_ACTIVE_KEYS": str(runtime["pressure_active_keys"]),
                "ORBITKV_PRESSURE_WINDOW_SHIFT": str(runtime["pressure_window_shift"]),
            },
        )
        log_path = artifact / "pressure-source.log"
        log = log_path.open("xb")
        process = subprocess.Popen(
            [
                str(self.runtime / "frozen/bin/server-tests"),
                PRESSURE_TEST,
                "--exact",
                "--ignored",
                "--nocapture",
            ],
            env=environment,
            cwd=self.output,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.pressure = process
        self.pressure_log = log
        self.pressure_output = artifact
        record = {
            "run": self.pressure_run,
            "pid": process.pid,
            "start_ticks": start_ticks(process.pid),
            "rounds": rounds,
            "status": "starting",
            "ready": None,
            "network_environment": direct_network_environment_evidence(
                environment, self.settings["network_environment"]
            ),
        }
        self.state["pressure_runs"].append(record)
        self.save_state()
        deadline = time.monotonic() + 30
        ready_path = artifact / "pressure-source-ready.json"
        while not ready_path.exists():
            if process.poll() is not None:
                raise RuntimeError(log_path.read_text(errors="replace"))
            if time.monotonic() >= deadline:
                raise TimeoutError("pressure source did not become ready")
            time.sleep(0.05)
        ready = json.loads(ready_path.read_text())
        go.touch()
        record.update({"status": "running", "ready": ready})
        self.save_state()
        return record

    def wait_pressure_complete(self) -> dict:
        result_path = self.pressure_output / "pressure-source-result.json"
        deadline = time.monotonic() + self.settings["budgets"]["phase_seconds"]
        while not result_path.exists():
            if self.pressure.poll() is not None:
                raise RuntimeError(
                    (self.pressure_output / "pressure-source.log").read_text(errors="replace")
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("pressure schedule did not complete")
            time.sleep(0.1)
        result = json.loads(result_path.read_text())
        self.last_pressure_result = result
        return result

    def stop_pressure(self, reason: str) -> dict:
        if self.pressure is None:
            return {"status": "not_started", "reason": reason}
        cleanup_path = self.pressure_output / "pressure-cleanup.json"
        if cleanup_path.exists():
            previous = json.loads(cleanup_path.read_text())
            if previous["status"] == "quarantined":
                return previous
        process = self.pressure
        preexisting = process.poll()
        if preexisting is not None:
            returncode = process.wait()
            record = {
                "status": "preexisting_exit",
                "reason": reason,
                "returncode": returncode,
                "reaped": True,
                "stop_requested": False,
            }
            atomic_json(cleanup_path, record)
            raise RuntimeError(f"pressure source exited before stop request: {record}")
        (self.pressure_output / "pressure.stop").touch(exist_ok=True)
        try:
            returncode = process.wait(timeout=self.cleanup_remaining(30))
        except subprocess.TimeoutExpired:
            record = {
                "status": "quarantined",
                "reason": reason,
                "pid": process.pid,
                "start_ticks": start_ticks(process.pid),
            }
            atomic_json(cleanup_path, record)
            return record
        self.pressure_log.close()
        record = {
            "status": "stopped" if returncode == 0 else "failed",
            "reason": reason,
            "returncode": returncode,
            "reaped": True,
            "stop_requested": True,
        }
        atomic_json(cleanup_path, record)
        self.pressure = None
        self.pressure_log = None
        if returncode != 0:
            raise RuntimeError(f"pressure source exit={returncode}: {record}")
        return record

    def snapshot(self, phase: str) -> dict:
        pressure_poll = self.pressure.poll() if self.pressure is not None else None
        return {
            "phase": phase,
            "metadata": self.metadata(),
            "owners": self.owners(),
            "pressure_poll": pressure_poll,
            "observed_unix_ns": time.time_ns(),
        }

    def start_cold_observer(self) -> dict:
        preexisting_member_absence = self.wait_member_absent("cold-observer", None)
        observer = self.start_manager(observer=True)
        self.cold_observer_manager = observer
        deadline = time.monotonic() + 30
        while True:
            metadata = self.metadata(observer)
            owners = self.owners(observer)
            if metadata["index"]["registration_valid"] and owners:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("cold observer did not converge")
            time.sleep(0.1)
        member_evidence = self.wait_etcd_member("cold-observer")
        self.cold_observer_member = member_evidence
        pid = observer.process.pid
        ticks = start_ticks(pid)
        runtime_owner = next(
            (
                record
                for record in self.state["manager_owners"]
                if record.get("node") == "cold-observer" and record.get("pid") == pid
            ),
            None,
        )
        if (
            observer.process.poll() is not None
            or not isinstance(ticks, str)
            or not ticks
            or runtime_owner is None
            or runtime_owner.get("status") != "running"
            or runtime_owner.get("start_ticks") != ticks
        ):
            raise RuntimeError(
                f"cold observer runtime ownership differs: {runtime_owner}, {pid}, {ticks}"
            )
        return {
            "metadata": metadata,
            "owners": owners,
            "member_evidence": member_evidence,
            "preexisting_member_absence": preexisting_member_absence,
            "runtime_owner": {
                **runtime_owner,
                "peer_endpoint": (
                    f"127.0.0.1:{self.settings['orbitkv_runtime']['observer_peer_port']}"
                ),
                "process_live": True,
            },
        }

    def stop_cold_observer(self) -> dict:
        observer = self.cold_observer_manager
        member_evidence = self.cold_observer_member
        if observer is None or member_evidence is None:
            raise RuntimeError("cold observer is not owned")
        try:
            termination = self._terminate_manager_gracefully(observer, timeout=10)
        except BaseException as error:
            cleanup = self._stop_manager_owner(observer, timeout=5)
            atomic_json(
                self.output / "cold-observer-cleanup.json",
                {
                    "status": cleanup.get("status", "error"),
                    "error": repr(error),
                    "manager_cleanup": cleanup,
                },
            )
            raise
        cleanup = self._stop_manager_owner(observer, timeout=5)
        result = {
            "termination": termination,
            "manager_exit": termination["returncode"],
            "shutdown_seconds": termination["shutdown_seconds"],
            "manager_cleanup": cleanup,
        }
        if (
            termination["status"] != "stopped"
            or termination["returncode"] != 0
            or cleanup["status"] not in ("exited", "already_released")
            or cleanup.get("returncode", 0) not in (None, 0)
        ):
            atomic_json(self.output / "cold-observer-cleanup.json", result)
            raise RuntimeError(f"cold observer cleanup is unresolved: {result}")
        result["member_absence"] = self.wait_member_absent("cold-observer", member_evidence)
        self.cold_observer_manager = None
        self.cold_observer_member = None
        atomic_json(self.output / "cold-observer-cleanup.json", result)
        return result

    def stop_query_source(self, reason: str, *, require_transfer_metric: bool) -> dict:
        import torch

        if self.query_source_manager is None:
            return {"status": "not_started", "reason": reason}
        ok, message = self.query_source_client.unregister_context("query-source")
        if not ok:
            raise RuntimeError(message)
        self.query_source_client.close()
        self.query_source_client = None
        metrics = fetch_orbitkv_metrics(self.query_source_manager.http_port)
        transfer_metric = "orbitkv_transfer_lock_active"
        if require_transfer_metric and transfer_metric not in metrics:
            raise RuntimeError("query source transfer-lock metric is absent")
        if transfer_metric not in metrics:
            raise RuntimeError("query source transfer-lock metric is absent")
        value = metrics[transfer_metric]
        if type(value) not in (int, float) or not math.isfinite(value) or value != 0:
            raise RuntimeError(f"query source transfer lock did not drain: {metrics}")
        termination = self._terminate_manager_gracefully(self.query_source_manager, timeout=10)
        cleanup = self._stop_manager_owner(self.query_source_manager, timeout=5)
        result = {
            "status": "stopped",
            "reason": reason,
            "termination": termination,
            "manager_cleanup": cleanup,
            "drain": metrics,
        }
        if (
            termination["status"] != "stopped"
            or termination["returncode"] != 0
            or cleanup["status"] != "already_released"
        ):
            raise RuntimeError(f"query source cleanup did not finish: {result}")
        self.query_source_client = None
        self.query_source_manager = None
        self.query_source_tensor = None
        self.query_source_hashes = None
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        return result

    def recover_after_quorum(self) -> dict:
        import torch

        old_members = dict(self.current_members)
        if set(old_members) != {"a100-consumer", "metadata-pressure", "query-source"}:
            raise RuntimeError(f"prior member identity is incomplete: {old_members}")
        old_pressure = self.stop_pressure("quorum_recovery_restart")
        old_query_source = self.stop_query_source(
            "quorum_recovery_restart", require_transfer_metric=False
        )
        ok, message = self.client.unregister_context("local")
        if not ok:
            raise RuntimeError(message)
        self.client.close()
        self.client = None
        drained = self.drain_metrics()
        termination = self._terminate_manager_gracefully(self.manager, timeout=10)
        code = termination["returncode"]
        seconds = termination["shutdown_seconds"]
        manager_cleanup = self._stop_manager_owner(self.manager, timeout=5)
        atomic_json(self.output / "quorum-recovery-manager-cleanup.json", manager_cleanup)
        if (
            termination["status"] != "stopped"
            or code != 0
            or manager_cleanup["status"] != "already_released"
        ):
            raise RuntimeError(f"old Manager recovery shutdown failed: {code}, {manager_cleanup}")
        old_member_absence = {
            node: self.wait_member_absent(node, old_members[node]) for node in old_members
        }
        self.tensor = None
        self.expected = None
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        local = self.start_local_cache()
        query_source = self.start_query_source()
        pressure = self.start_pressure(1200)
        self.current_members = self.capture_current_members(pressure["ready"])
        self.state.update(
            {
                "manager_pid": self.manager.process.pid,
                "manager_start_ticks": start_ticks(self.manager.process.pid),
                "local": local,
                "query_source": query_source,
                "pressure": pressure,
                "members": self.current_members,
            }
        )
        self.save_state()
        return {
            "old_pressure": old_pressure,
            "old_members": old_members,
            "old_member_absence": old_member_absence,
            "old_query_source": old_query_source,
            "old_manager_exit": code,
            "old_manager_shutdown_seconds": seconds,
            "old_manager_termination": termination,
            "old_manager_cleanup": manager_cleanup,
            "old_manager_drain": drained,
            "local": local,
            "query_source": query_source,
            "pressure": pressure,
            "members": self.current_members,
        }

    def initialize(self) -> None:
        self.output.mkdir(parents=True, exist_ok=False)
        self.controls.mkdir()
        self.responses.mkdir()
        lock = (self.output / ".agent.lock").open("x")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state["preflight"] = self.preflight()
        self.save_state()
        local = self.start_local_cache()
        query_source = self.start_query_source()
        pressure = self.start_pressure(self.settings["orbitkv_runtime"]["pressure_rounds"])
        self.current_members = self.capture_current_members(pressure["ready"])
        self.state.update(
            {
                "state": "running",
                "manager_pid": self.manager.process.pid,
                "manager_start_ticks": start_ticks(self.manager.process.pid),
                "local": local,
                "query_source": query_source,
                "pressure": pressure,
                "members": self.current_members,
            }
        )
        self.save_state()

    def cleanup(self) -> dict:
        if self.cleanup_deadline is None:
            self.cleanup_deadline = time.monotonic() + self.settings["budgets"]["cleanup_seconds"]
        self.cleanup_remaining(0.001)
        import torch

        if self.cold_observer_manager is not None:
            raise RuntimeError("cold observer still owns a live Manager during final cleanup")
        pressure = self.stop_pressure("final_agent_exit")
        if pressure["status"] not in ("stopped", "not_started"):
            raise RuntimeError(f"pressure cleanup did not finish: {pressure}")
        query_source = self.stop_query_source("final_agent_exit", require_transfer_metric=True)
        ok, message = self.client.unregister_context("local")
        if not ok:
            raise RuntimeError(message)
        self.client.close()
        self.client = None
        drained = self.drain_metrics()
        termination = self._terminate_manager_gracefully(self.manager, timeout=10)
        code = termination["returncode"]
        seconds = termination["shutdown_seconds"]
        manager_cleanup = self._stop_manager_owner(self.manager, timeout=5)
        result = {
            "pressure": pressure,
            "query_source": query_source,
            "manager_exit": code,
            "manager_shutdown_seconds": seconds,
            "manager_cleanup": manager_cleanup,
            "manager_drain": drained,
            "manager_termination": termination,
        }
        if (
            termination["status"] != "stopped"
            or code != 0
            or manager_cleanup["status"] != "already_released"
        ):
            raise RuntimeError(f"Manager cleanup did not finish: {result}")
        self.tensor = None
        self.expected = None
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        uds_cleanup = self.release_uds_root()
        result["uds_root_cleanup"] = uds_cleanup
        if uds_cleanup["status"] != "released":
            raise RuntimeError(f"Manager UDS cleanup did not finish: {uds_cleanup}")
        return result

    def cleanup_partial(self, reason: str) -> dict:
        if self.cleanup_deadline is None:
            self.cleanup_deadline = time.monotonic() + self.settings["budgets"]["cleanup_seconds"]
        result = {
            "reason": reason,
            "pressure": None,
            "client": None,
            "query_source_client": None,
            "manager": None,
            "managers": [],
        }
        if self.pressure is not None:
            try:
                result["pressure"] = self.stop_pressure(reason)
            except BaseException as error:
                result["pressure"] = {"status": "error", "error": repr(error)}
        client_safe = self.client is None and self.query_source_client is None
        if self.client is not None:
            try:
                ok, message = self.client.unregister_context("local")
                result["client"] = {"ok": ok, "message": message}
                if not ok:
                    raise RuntimeError(message)
                self.client.close()
                self.client = None
                client_safe = self.query_source_client is None
            except BaseException as error:
                client_safe = False
                result["client"] = {"status": "error", "error": repr(error)}
        if self.query_source_client is not None:
            try:
                ok, message = self.query_source_client.unregister_context("query-source")
                result["query_source_client"] = {"ok": ok, "message": message}
                if not ok:
                    raise RuntimeError(message)
                self.query_source_client.close()
                self.query_source_client = None
                client_safe = self.client is None
            except BaseException as error:
                client_safe = False
                result["query_source_client"] = {
                    "status": "error",
                    "error": repr(error),
                }
        for manager in list(self.manager_owners):
            cleanup = self._stop_manager_owner(manager, timeout=5)
            result["managers"].append(cleanup)
            if manager is self.manager:
                result["manager"] = cleanup
        result["failed_managers"] = list(self.failed_manager_cleanups)
        manager_safe = not self.manager_owners and all(
            record.get("status") in ("exited", "already_released")
            and record.get("returncode", 0) in (None, 0)
            for record in result["managers"]
        )
        if self.tensor is not None and manager_safe and client_safe:
            import torch

            self.tensor = None
            self.expected = None
            self.query_source_tensor = None
            self.query_source_hashes = None
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            result["tensor_released"] = True
        else:
            result["tensor_released"] = self.tensor is None and self.query_source_tensor is None
        pressure_safe = result["pressure"] is None or result["pressure"].get("status") in (
            "stopped",
            "not_started",
        )
        ownership_safe = (
            pressure_safe and client_safe and manager_safe and result["tensor_released"]
        )
        if ownership_safe:
            uds_cleanup = self.release_uds_root()
        else:
            uds_cleanup = {
                "status": "retained_after_uncertain_cleanup",
                "owner": self.uds_root_owner,
            }
        result["uds_root_cleanup"] = uds_cleanup
        result["quarantined"] = not (
            ownership_safe and uds_cleanup["status"] in {"released", "not_created"}
        )
        return result

    def run(self) -> None:
        try:
            self.initialize()
            while True:
                requests_paths = sorted(self.controls.glob("*.request"))
                for request in requests_paths:
                    stem = request.name.removesuffix(".request")
                    if stem in self.processed:
                        continue
                    sequence, action = stem.split("-", 1)
                    request_budget = json.loads(request.read_text() or "{}").get(
                        "cleanup_remaining_seconds"
                    )
                    self.accept_cleanup_budget(request_budget)
                    try:
                        if action.startswith("local_"):
                            result = self.check_local(action.removeprefix("local_"))
                        elif action.startswith("snapshot_"):
                            result = self.snapshot(action.removeprefix("snapshot_"))
                        elif action == "wait_pressure":
                            result = self.wait_pressure_complete()
                        elif action == "replacement":
                            old_cleanup = self.stop_pressure("lease_revoked_replacement")
                            if old_cleanup["status"] not in ("stopped", "not_started"):
                                raise RuntimeError(f"old pressure cleanup failed: {old_cleanup}")
                            replacement = self.start_pressure(600)
                            member = self.wait_etcd_member(
                                "metadata-pressure",
                                expected_owner=replacement["ready"]["owner"],
                            )
                            self.current_members["metadata-pressure"] = member
                            result = {
                                "old_cleanup": old_cleanup,
                                "replacement": replacement,
                                "member": member,
                            }
                        elif action == "recover":
                            result = self.recover_after_quorum()
                        elif action == "cold_observer_start":
                            result = self.start_cold_observer()
                        elif action == "cold_observer_stop":
                            result = self.stop_cold_observer()
                        elif action == "query_source_present":
                            result = self.query_readable_source(expected_hits=1)
                        elif action == "prepare_query_source_absent":
                            result = self.prepare_stale_source_query()
                        elif action == "query_source_absent":
                            result = self.query_readable_source(expected_hits=0)
                        elif action == "exit":
                            result = self.cleanup()
                        else:
                            raise RuntimeError(f"unsupported action: {action}")
                    except BaseException as error:
                        atomic_json(
                            self.responses / f"{stem}.json",
                            {
                                "sequence": sequence,
                                "action": action,
                                "error": repr(error),
                            },
                        )
                        self.processed.append(stem)
                        self.state["action_error"] = {
                            "action": action,
                            "error": repr(error),
                        }
                        self.save_state()
                        raise
                    atomic_json(
                        self.responses / f"{stem}.json",
                        {"sequence": sequence, "action": action, "result": result},
                    )
                    self.processed.append(stem)
                    if action == "exit":
                        self.state["state"] = "exited"
                        self.state["cleanup"] = result
                        self.save_state()
                        return
                    self.save_state()
                time.sleep(0.1)
        except BaseException as error:
            self.state["state"] = "invalid"
            self.state["error"] = repr(error)
            cleanup = self.cleanup_partial("consumer_agent_failure")
            self.state["failure_cleanup"] = cleanup
            if cleanup["quarantined"]:
                self.state["state"] = "timeout_quarantined"
            self.save_state()
            atomic_json(self.output / "failure-cleanup.json", cleanup)
            if cleanup["quarantined"]:
                while True:
                    time.sleep(60)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--approved-input-manifest-sha256", required=True)
    args = parser.parse_args()
    settings = json.loads((args.root / "inputs/settings.json").read_text())
    direct_network_environment_evidence(os.environ, settings["network_environment"])
    Agent(args.root, settings, args.approved_input_manifest_sha256).run()


if __name__ == "__main__":
    main()
