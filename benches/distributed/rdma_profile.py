"""Run a reviewed two-host TENT GPU and full-Manager descriptive profile.

Each host retains its inherited leases if control or native termination becomes
unknown. Failed cells remain in the cohort and stop new admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import statistics
import subprocess
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import requests

from benches.common.artifacts import external_path
from benches.distributed.shared_cache import drain
from benches.distributed.shared_cache_profile import profile
from benches.distributed.tent_gpu import CAPACITY


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".next")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def statistics_for(values: list[float]) -> dict:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Timing samples must be positive and finite")
    return {
        "p50": statistics.median(values),
        "p95_nearest_rank": sorted(values)[math.ceil(len(values) * 0.95) - 1],
        "min": min(values),
        "max": max(values),
    }


def nic_delta(before: dict, after: dict) -> dict:
    if before["boot_id"] != after["boot_id"]:
        raise RuntimeError("Host boot changed during a cell")
    result = {}
    for nic in before["nics"]:
        result[nic] = {}
        for counter in ("port_xmit_data", "port_rcv_data", "port_xmit_packets", "port_rcv_packets"):
            difference = after["nics"][nic][counter] - before["nics"][nic][counter]
            if difference < 0:
                raise RuntimeError("Physical NIC counters reset")
            result[nic][counter] = difference
        result[nic]["tx_bytes"] = result[nic]["port_xmit_data"] * 4
        result[nic]["rx_bytes"] = result[nic]["port_rcv_data"] * 4
    return result


class HostConnection:
    """Own one SSH controller and consume the remote owner's matched receipts."""

    def __init__(self, command: list[str], host: str, output: Path):
        self.host, self.output, self.sequence = host, output, 0
        self.command = command
        self.process = None
        self.thread = None
        self.protocol = queue.Queue(maxsize=64)
        self.uncertain = self.finished = False
        self.stdout = (output / (host + ".stdout.log")).open("x")
        self.stderr = (output / (host + ".stderr.log")).open("x")

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )

        def read():
            try:
                for line in self.process.stdout:
                    self.stdout.write(line)
                    self.stdout.flush()
                    value = json.loads(line)
                    if value.get("rdma_owner") is not True:
                        raise RuntimeError("Unexpected owner protocol")
                    self.protocol.put(value, timeout=5)
            except Exception as error:
                self.protocol.put({"transport_error": str(error)}, timeout=5)
            finally:
                self.protocol.put({"transport_error": "SSH owner EOF"}, timeout=5)

        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()
        value = self.response()
        if value.get("host") != self.host or value.get("ready") is not True:
            self.uncertain = True
            raise RuntimeError(("Owner not ready", value))
        try:
            save(
                self.output / (self.host + "-owner-ready.json"),
                {"argv": self.command, "receipt": value},
            )
        except Exception:
            self.uncertain = True
            raise

    def response(self) -> dict:
        try:
            value = self.protocol.get(timeout=100)
            if "transport_error" in value:
                raise RuntimeError(value)
            return value
        except Exception:
            self.uncertain = True
            raise

    def rpc(self, operation: str, **fields) -> dict:
        if self.uncertain or self.finished:
            raise RuntimeError("Unknown/finished owner cannot admit work")
        self.sequence += 1
        request_id = f"{self.host}-{self.sequence}"
        try:
            self.process.stdin.write(
                json.dumps({"operation": operation, "id": request_id, **fields}) + "\n"
            )
            self.process.stdin.flush()
            response = self.response()
        except Exception:
            self.uncertain = True
            raise
        if response.get("id") != request_id:
            self.uncertain = True
            raise RuntimeError("Owner response id mismatch")
        if response.get("ok") is not True:
            self.uncertain |= response.get("ownership_uncertain") is True
            raise RuntimeError(response)
        return response["result"]

    def finish(self) -> dict:
        if self.process is None:
            self.finished = True
            self.stdout.close()
            self.stderr.close()
            return {"process_not_started": True, "no_remote_work_admitted": True}
        result = self.rpc("finish")
        self.finished = True
        code = self.process.wait(timeout=30)
        self.thread.join(timeout=5)
        if code or self.thread.is_alive():
            raise RuntimeError(("Owner failed actual exit", code))
        for stream in (self.process.stdin, self.process.stdout, self.stdout, self.stderr):
            stream.close()
        return {"physical_postflight": result, "owner_exit_code": code, "actual_wait_reap": True}


class Campaign:
    def __init__(self, contract: dict, output: Path):
        self.contract, self.output = contract, output
        self.hosts = {}
        self.rows = []
        self.command_id = 0
        self.active = {}

    def launch(self, host: str, cell: str, name: str) -> dict:
        planned = self.contract["launches"][host][cell][name]
        receipt = self.hosts[host].rpc(
            "launch", cell=cell, name=name, argv=planned["argv"], environment=planned["environment"]
        )
        self.active.setdefault(host, {})[name] = cell
        return receipt

    def stop(self, host: str, name: str, **fields) -> dict:
        receipt = self.hosts[host].rpc("stop", name=name, **fields)
        del self.active[host][name]
        return receipt

    def gpu(self, host: str, name: str, operation: str, **fields) -> dict:
        self.command_id += 1
        value = self.hosts[host].rpc(
            "call",
            name=name,
            request={"operation": operation, "id": str(self.command_id), **fields},
        )
        if operation != "stop" and value.get("ok") is not True:
            raise AssertionError(value)
        return value

    def gpu_cell(self, cell: dict, output: Path) -> dict:
        reader, peer = cell["reader_host"], cell["peer_host"]
        before = {host: connection.rpc("snapshot") for host, connection in self.hosts.items()}
        launched = []
        result = {"cell": cell, "before": before, "samples": []}
        failed = None
        ready = {}
        try:
            for host, name in [(peer, "gpu-peer"), (reader, "gpu-reader")]:
                self.launch(host, cell["id"], name)
                launched.append((host, name))
                ready[host] = self.hosts[host].rpc("call", name=name)
                receipt = ready[host]
                if receipt.get("capacity") != CAPACITY or len(receipt.get("addresses", [])) != 1:
                    raise AssertionError("GPU registration geometry differs")
                if any(
                    row["gpu"] != 7
                    or row["memory_type"] != 2
                    or row["uuid"] != self.contract["hosts"][host]["gpu_uuids"]["7"]
                    for row in receipt["attributes"]
                ):
                    raise AssertionError("Expected physical GPU-7 device pointers")
                for pointer in receipt["addresses"]:
                    if type(pointer) is not int or pointer <= 0:
                        raise AssertionError("Invalid registered GPU pointer")
                self.gpu(host, name, "fill", seed=17, payload=CAPACITY)
                control = self.gpu(host, name, "oracle-control", seed=17, payload=CAPACITY)
                result.setdefault("negative_controls", {})[host] = control
            result["ready"] = ready
            source, target = (peer, reader) if cell["operation"] == "read" else (reader, peer)
            names = {reader: "gpu-reader", peer: "gpu-peer"}
            geometry = {key: cell[key] for key in ("payload", "chunk", "stride")}
            for iteration in range(1 + self.contract["warmup"] + self.contract["samples"]):
                seed = 1009 * (cell["ordinal"] + 1) + iteration
                self.gpu(source, names[source], "fill", seed=seed, payload=CAPACITY)
                self.gpu(target, names[target], "clear", **geometry)
                transferred = self.gpu(
                    reader,
                    "gpu-reader",
                    cell["operation"],
                    endpoint=ready[peer]["endpoint"],
                    remote_address=ready[peer]["addresses"][0],
                    **geometry,
                )
                checks = {
                    "source": self.gpu(
                        source, names[source], "verify", seed=seed, payload=CAPACITY
                    ),
                    "target": self.gpu(target, names[target], "verify", seed=seed, **geometry),
                }
                duration = transferred["binding_to_terminal_ms"]
                if not math.isfinite(duration) or duration <= 0:
                    raise AssertionError("Invalid terminal timing")
                phase = (
                    "cold"
                    if iteration == 0
                    else "warmup"
                    if iteration <= self.contract["warmup"]
                    else "measured"
                )
                result["samples"].append(
                    {
                        "iteration": iteration,
                        "phase": phase,
                        "seed": seed,
                        "payload_gb_s": cell["payload"] / (duration * 1e6),
                        "transfer": transferred,
                        "oracles": checks,
                    }
                )
                save(output, result)
        except Exception as error:
            failed = error
            result.update(status="INVALID_OR_FAILED", error=str(error))
        finally:
            if launched and all(not owner.uncertain for owner in self.hosts.values()):
                if (reader, "gpu-reader") in launched:
                    self.gpu(reader, "gpu-reader", "stop")
                    reader_exit = self.stop(reader, "gpu-reader")
                    result["reader_exit"] = reader_exit
                    if (peer, "gpu-peer") in launched:
                        self.gpu(peer, "gpu-peer", "stop", peer_receipt=reader_exit)
                        result["peer_exit"] = self.stop(peer, "gpu-peer")
                elif (peer, "gpu-peer") in launched:
                    raise RuntimeError("Peer export lacks an actual requester drain receipt")
            save(output, result)
        if failed:
            raise failed
        if result["reader_exit"]["exit_code"] != 0 or result["peer_exit"]["exit_code"] != 0:
            raise RuntimeError("GPU endpoint normal exit failed")
        after = {host: connection.rpc("snapshot") for host, connection in self.hosts.items()}
        measured = [row for row in result["samples"] if row["phase"] == "measured"]
        result.update(
            status="PASS_DESCRIPTIVE_PROFILE",
            after=after,
            nic_deltas={host: nic_delta(before[host], after[host]) for host in before},
            summary={
                "binding_to_terminal_ms": statistics_for(
                    [row["transfer"]["binding_to_terminal_ms"] for row in measured]
                ),
                "payload_gb_s": statistics_for([row["payload_gb_s"] for row in measured]),
            },
        )
        source = cell["peer_host"] if cell["operation"] == "read" else cell["reader_host"]
        target = cell["reader_host"] if cell["operation"] == "read" else cell["peer_host"]
        transferred_bytes = cell["payload"] * len(result["samples"])
        for host, direction in ((source, "tx_bytes"), (target, "rx_bytes")):
            if (
                sum(result["nic_deltas"][host][nic][direction] for nic in cell["nics"])
                < transferred_bytes
            ):
                raise AssertionError("Selected physical RDMA NIC counters do not cover payload")
        save(output, result)
        return result

    def manager_cell(self, cell: dict, output: Path) -> dict:
        before = {host: owner.rpc("snapshot") for host, owner in self.hosts.items()}
        urls = [cell["manager_urls"][host] for host in self.hosts]
        result = {"cell": cell, "before": before}
        launched = []
        error = None
        cleanup_error = None
        try:
            for name in ("manager", "fixture"):
                for host in self.hosts:
                    self.launch(host, cell["id"], name)
                    launched.append((host, name))
                    url = (
                        cell["manager_urls"][host]
                        if name == "manager"
                        else cell["fixture_urls"][host]
                    )
                    ports = [urlsplit(url).port]
                    if name == "manager":
                        argv = self.contract["launches"][host][cell["id"]][name]["argv"]
                        ports.append(urlsplit("tcp://" + argv[argv.index("--addr") + 1]).port)
                    result.setdefault("readiness", []).append(self.health(host, name, url, ports))
            result["runtime_admission"] = {
                host: owner.rpc("admit-runtime", cell=cell["id"])
                for host, owner in self.hosts.items()
            }
            source, target = cell["source_host"], cell["target_host"]
            evidence = profile(
                SimpleNamespace(
                    source_url=cell["fixture_urls"][source],
                    target_url=cell["fixture_urls"][target],
                    source_manager=cell["manager_urls"][source],
                    target_manager=cell["manager_urls"][target],
                    payload_bytes=cell["payload"],
                    warmup=self.contract["warmup"],
                    samples=self.contract["samples"],
                    output=output.with_suffix(".samples.json"),
                )
            )
            result["evidence"] = evidence
            quiet = drain(*urls)
            for row in quiet:
                for key in self.contract["required_idle_metrics"]:
                    if row.get(key) != 0:
                        raise AssertionError(
                            ("Missing/nonzero physical-owner gauge", key, row.get(key))
                        )
            result["pre_stop_idle"] = quiet
        except Exception as caught:
            error = caught
            result.update(
                status="INVALID_OR_FAILED",
                error=str(caught),
                error_type=type(caught).__name__,
                traceback=traceback.format_exc(),
            )
            save(output, result)
        finally:
            try:
                if launched and all(not owner.uncertain for owner in self.hosts.values()):
                    payload_admitted = any(name == "fixture" for _, name in launched)
                    if payload_admitted:
                        for host, name in launched:
                            if name == "fixture":
                                url = cell["fixture_urls"][host]
                                self.health(host, name, url, [urlsplit(url).port])
                        quiet = drain(*urls)
                        for row in quiet:
                            if any(
                                row.get(key) != 0 for key in self.contract["required_idle_metrics"]
                            ):
                                raise RuntimeError(
                                    "No complete owner drain proof before service stop"
                                )
                        result["cleanup_idle"] = quiet
                    else:
                        result["cleanup_proof"] = (
                            "No fixture/payload actor was admitted in this fresh cell"
                        )
                    result["exits"] = []
                    for host, name in reversed(launched):
                        receipt = self.stop(host, name, idle_proven=True)
                        result["exits"].append(receipt)
                        expected = 143 if name == "fixture" else 0
                        if error is None and receipt["exit_code"] != expected:
                            raise RuntimeError(
                                ("Unexpected normal service exit", expected, receipt)
                            )
                    self.members_absent(cell)
            except Exception as caught:
                cleanup_error = caught
                result.update(
                    status="INVALID_OR_FAILED",
                    cleanup_error=str(caught),
                    cleanup_traceback=traceback.format_exc(),
                )
            finally:
                save(output, result)
        if error:
            raise error
        if cleanup_error:
            raise cleanup_error
        after = {host: owner.rpc("snapshot") for host, owner in self.hosts.items()}
        result.update(
            status="PASS_DESCRIPTIVE_PROFILE",
            after=after,
            nic_deltas={host: nic_delta(before[host], after[host]) for host in before},
        )
        save(output, result)
        return result

    def health(self, host: str, name: str, url: str, ports: list[int]) -> dict:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state = self.hosts[host].rpc("process-status", name=name)
            if state["exit_code"] is not None:
                raise ChildProcessError(("Owned service exited before readiness", state))
            try:
                if (
                    set(ports) <= set(state["listen_ports"])
                    and requests.get(url + "/health", timeout=3).status_code == 200
                ):
                    after = self.hosts[host].rpc("process-status", name=name)
                    if after["exit_code"] is not None:
                        raise ChildProcessError(("Owned service exited during readiness", after))
                    if set(ports) <= set(after["listen_ports"]):
                        return after
            except requests.RequestException:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"Service not healthy: {url}")

    def members_absent(self, cell: dict) -> None:
        import base64

        prefix = cell["member_prefix"].encode()
        upper = prefix[:-1] + bytes([prefix[-1] + 1])
        response = requests.post(
            self.contract["etcd_url"] + "/v3/kv/range",
            timeout=10,
            json={
                "key": base64.b64encode(prefix).decode(),
                "range_end": base64.b64encode(upper).decode(),
            },
        )
        response.raise_for_status()
        if response.json().get("kvs"):
            raise AssertionError("Membership remains after normal Manager stop")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=external_path, required=True)
    args = parser.parse_args()
    inputs = (args.contract.parent / "INPUTS.json").read_bytes()
    receipt = json.loads((args.contract.parent / "LAUNCH-ACCEPTED.json").read_bytes())
    if (
        receipt.get("accepted") is not True
        or receipt.get("inputs_sha256") != hashlib.sha256(inputs).hexdigest()
    ):
        raise RuntimeError("Exact cohort lacks independent launch acceptance")
    contract_bytes = args.contract.read_bytes()
    if hashlib.sha256(contract_bytes).hexdigest() != json.loads(inputs).get(args.contract.name):
        raise RuntimeError("Contract differs from reviewed inputs")
    contract = json.loads(contract_bytes)
    for relative, digest in contract["source_hashes"].items():
        if (
            hashlib.sha256(
                (Path(__file__).resolve().parents[2] / relative).read_bytes()
            ).hexdigest()
            != digest
        ):
            raise RuntimeError(f"Controller source changed: {relative}")
    for name in list(os.environ):
        if name.lower() in ("http_proxy", "https_proxy", "all_proxy"):
            del os.environ[name]
    if args.output.exists():
        raise ValueError("A cohort output directory may not be reused")
    args.output.mkdir(parents=True)
    campaign = Campaign(contract, args.output)
    status = {
        "status": "RUNNING",
        "completed_cells": 0,
        "cell_count": len(contract["cells"]),
        "cells": [],
    }
    save(args.output / "STATUS.json", status)
    try:
        for host, command in contract["owner_commands"].items():
            campaign.hosts[host] = HostConnection(command, host, args.output)
            campaign.hosts[host].start()
        if any(cell["kind"] == "manager" for cell in contract["cells"]):
            campaign.launch("h0", "campaign", "etcd")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    if requests.get(contract["etcd_url"] + "/health", timeout=3).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError("Private etcd not ready")
        for cell in contract["cells"]:
            status["active_cell"] = cell["id"]
            save(args.output / "STATUS.json", status)
            runner = campaign.gpu_cell if cell["kind"] == "gpu" else campaign.manager_cell
            result = runner(cell, args.output / (cell["id"] + ".json"))
            status["cells"].append({"id": cell["id"], "status": result["status"]})
            status["completed_cells"] += 1
            save(args.output / "STATUS.json", status)
        if any(cell["kind"] == "manager" for cell in contract["cells"]):
            etcd_exit = campaign.stop("h0", "etcd", idle_proven=True)
            if etcd_exit["exit_code"] != -15:
                raise RuntimeError(("Private etcd did not exit from SIGTERM", etcd_exit))
            status["etcd_exit"] = etcd_exit
        status["postflight"] = {host: owner.finish() for host, owner in campaign.hosts.items()}
        status.update(
            status="PASS_DESCRIPTIVE_PROFILE",
            active_cell=None,
            qualified_support_changed=False,
            production_strategy_changed=False,
        )
    except Exception as error:
        status.update(
            status="INVALID_OR_FAILED_FAIL_STOP",
            error=str(error),
            ownership_uncertain={host: owner.uncertain for host, owner in campaign.hosts.items()},
            owners={
                host: {
                    "argv": owner.command,
                    "ssh_pid": owner.process.pid if owner.process else None,
                    "ssh_returncode": owner.process.returncode if owner.process else None,
                    "finished": owner.finished,
                }
                for host, owner in campaign.hosts.items()
            },
        )
        if all(not owner.uncertain for owner in campaign.hosts.values()) and all(
            set(children) <= {"etcd"} for children in campaign.active.values()
        ):
            try:
                if "etcd" in campaign.active.get("h0", {}):
                    status["cleanup_etcd_exit"] = campaign.stop("h0", "etcd", idle_proven=True)
                status["failure_postflight"] = {
                    host: owner.finish()
                    for host, owner in campaign.hosts.items()
                    if not owner.finished
                }
            except Exception as cleanup_error:
                status["cleanup_error"] = str(cleanup_error)
        raise
    finally:
        save(args.output / "STATUS.json", status)


if __name__ == "__main__":
    main()
