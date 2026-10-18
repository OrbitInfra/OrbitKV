"""Compare frozen native TENT builds on two SSH-controlled RDMA hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import random
import shlex
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ssh(contract: dict, host: str, argv: list[str]) -> str:
    result = subprocess.run(
        ["ssh", "-F", contract["ssh_config"], host, shlex.join(argv)],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise RuntimeError((host, result.returncode, result.stdout, result.stderr))
    return result.stdout


def snapshot(contract: dict, host: str, pid: int | None = None) -> dict:
    code = (
        "import pathlib,hashlib,json,subprocess,os\n"
        f"dirs={json.dumps(contract['library_directories'])}\n"
        "r={'libraries':{k:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() "
        "for p in pathlib.Path(v).iterdir() if p.is_file()} for k,v in dirs.items()},"
        "'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps="
        "pid,process_name,used_memory','--format=csv,noheader'],text=True).strip(),"
        "'uname':subprocess.check_output(['uname','-a'],text=True).strip()}\n"
        f"r['rdma_counters']={{}}\nfor nic in {contract['nics']!r}:\n"
        " p=pathlib.Path('/sys/class/infiniband')/nic/'ports/1/counters'\n"
        " r['rdma_counters'][nic]={n:int((p/n).read_text()) for n in "
        "('port_xmit_data','port_rcv_data','port_xmit_packets','port_rcv_packets')}\n"
        "r['rdma_data_counter_unit_bytes']=4\n"
    )
    if pid is not None:
        code += (
            f"p=pathlib.Path('/proc/{pid}')\n"
            "s=p.joinpath('stat').read_text().rsplit(')',1)[1].split()\n"
            "r.update({'pid':p.name,'cpu_ticks':int(s[11])+int(s[12]),"
            "'start_ticks':int(s[19]),'clk_tck':os.sysconf('SC_CLK_TCK'),"
            "'status':p.joinpath('status').read_text(),"
            "'native_mappings':sorted(set(x.split()[-1] for x in "
            "p.joinpath('maps').read_text().splitlines() "
            "if any(v+'/' in x for v in dirs.values())))})\n"
        )
    return json.loads(ssh(contract, host, ["python3", "-B", "-c", code + "print(json.dumps(r))"]))


class Endpoint:
    def __init__(self, contract: dict, out: Path, host: dict, name: str, variant: str, role: str):
        self.contract, self.out = contract, out
        self.host, self.name, self.variant = host["alias"], name, variant
        self.stopped, self.ready = False, None
        self.events = queue.Queue()
        self.stdout = (out / (name + ".stdout")).open("x")
        self.stderr = (out / (name + ".stderr")).open("x")
        library = contract["library_directories"][variant]
        env = [
            "env",
            "-u",
            "MC_FORCE_TCP",
            "-u",
            "PYTHONPATH",
            "-u",
            "PYTHONOPTIMIZE",
            "-u",
            "LD_PRELOAD",
            "-u",
            "TENT_NATIVE_TRACE_PATH",
            "PYTHONDONTWRITEBYTECODE=1",
            "MC_TENT_CONF=" + contract["remote_config"],
            "MC_TE_FILTERS=" + ",".join(contract["nics"]),
            "MC_GID_INDEX=" + str(contract["gid_index"]),
            "LD_LIBRARY_PATH=" + library + ":/usr/local/cuda/lib64",
        ]
        command = [
            *env,
            "python3",
            "-B",
            contract["remote_probe"],
            "--bind",
            host["address"],
            "--native-lib",
            library,
            "--role",
            role,
            "--bytes",
            str(contract["design"]["payload_bytes"]),
        ]
        (out / (name + ".command.json")).write_text(json.dumps(command, indent=2) + "\n")
        self.process = subprocess.Popen(
            ["ssh", "-F", contract["ssh_config"], self.host, shlex.join(command)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self.read_stdout, daemon=True)
        self.thread.start()

    def start(self) -> None:
        self.ready = self.event("ready")
        self.before = snapshot(self.contract, self.host, self.ready["pid"])
        expected = self.contract["library_directories"][self.variant]
        paths = self.before["native_mappings"]
        for name in ("libtent_shared.so", "libmooncake_common.so", "libasio.so"):
            if expected + "/" + name not in paths:
                raise RuntimeError((self.name, "wrong native mapping", paths))

    def read_stdout(self) -> None:
        for line in self.process.stdout:
            self.stdout.write(line)
            self.stdout.flush()
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if value.get("tent_stage_probe"):
                self.events.put(value)
        self.events.put({"event": "eof"})

    def event(self, name: str) -> dict:
        deadline = time.monotonic() + 120
        while True:
            value = self.events.get(timeout=max(0.01, deadline - time.monotonic()))
            if value["event"] == name:
                return value
            raise RuntimeError((self.name, "unexpected event", name, value))

    def send(self, value: dict) -> None:
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def stop(self) -> dict:
        after = snapshot(self.contract, self.host, self.ready["pid"])
        self.send({"operation": "stop"})
        event = self.event("stopped")
        code = self.process.wait(timeout=120)
        self.thread.join(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.stdout.close()
        self.stderr.close()
        if code or event["registered_regions"] or event["active_batches"] or self.thread.is_alive():
            raise RuntimeError((self.name, "not normally drained", code, event))
        if self.before["start_ticks"] != after["start_ticks"]:
            raise RuntimeError("Endpoint PID was reused")
        log = (self.out / (self.name + ".stderr")).read_text()
        marker = "Loaded tent config from file: " + self.contract["remote_config"]
        if marker not in log:
            raise RuntimeError("Frozen native config was not consumed")
        ssh(
            self.contract,
            self.host,
            [
                "python3",
                "-B",
                "-c",
                f"import pathlib; assert not pathlib.Path('/proc/{self.ready['pid']}').exists()",
            ],
        )
        self.stopped = True
        return {
            "host": self.host,
            "name": self.name,
            "pid": self.ready["pid"],
            "exit_code": code,
            "forced_cleanup": False,
            "before": self.before,
            "after": after,
            "cpu_tick_delta": after["cpu_ticks"] - self.before["cpu_ticks"],
            "stopped": event,
        }


def geometric(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(x) for x in values))


def interval(values: list[float], seed: int, draws: int) -> list[float]:
    rng = random.Random(seed)
    ordered = sorted(geometric(rng.choices(values, k=len(values))) for _ in range(draws))
    return [ordered[math.ceil(draws * 0.025) - 1], ordered[math.ceil(draws * 0.975) - 1]]


def run(contract: dict, out: Path) -> None:
    owners, cells, pairs, summaries = [], [], [], []
    result = {"state": "INVALID_FAIL_STOP", "independent_acceptance": False}
    design, guards = contract["design"], contract["guards"]
    try:
        for host in contract["hosts"]:
            before = snapshot(contract, host["alias"])
            if before["gpu"] or before["libraries"] != contract["libraries"]:
                raise RuntimeError("Runtime changed or GPU is in use")
            (out / (host["alias"] + "-preflight.json")).write_text(
                json.dumps(before, indent=2) + "\n"
            )
            for name in ("probe", "config"):
                actual = ssh(contract, host["alias"], ["sha256sum", contract["remote_" + name]])
                if actual.split()[0] != contract["input_hashes"][name]:
                    raise RuntimeError("Frozen remote input changed: " + name)
        for direction in (0, 1):
            for pair in range(design["pairs_per_direction"]):
                order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
                rows = {}
                for variant in order:
                    name = f"d{direction}-p{pair}-{variant}"
                    endpoints = []
                    for side, role in ((direction, "source"), (1 - direction, "consumer")):
                        endpoint = Endpoint(
                            contract, out, contract["hosts"][side], name + "-" + role, variant, role
                        )
                        owners.append(endpoint)
                        endpoints.append(endpoint)
                        endpoint.start()
                    source, consumer = endpoints
                    if source.ready["sha256"] != consumer.ready["sha256"]:
                        raise RuntimeError("Endpoint payloads differ")
                    total = 1 + design["warmup"] + design["measured"]
                    consumer.send(
                        {
                            "operation": "read",
                            "id": name,
                            "endpoint": source.ready["endpoint"],
                            "address": source.ready["address"],
                            "iterations": total,
                        }
                    )
                    event = consumer.event("read_complete")
                    samples = event["samples"]
                    if (
                        event["id"] != name
                        or len(samples) != total
                        or not all(
                            s["checked_bytes"] == s["transferred_bytes"] == design["payload_bytes"]
                            and s["sha256"] == source.ready["sha256"]
                            for s in samples
                        )
                    ):
                        raise RuntimeError("READ oracle failed")
                    notify = {"id": name, "count": design["notifications"]}
                    source.send({"operation": "receive_notifications", **notify})
                    if source.event("receiving_notifications")["id"] != name:
                        raise RuntimeError("Wrong notification receiver")
                    consumer.send(
                        {
                            "operation": "send_notifications",
                            "endpoint": source.ready["endpoint"],
                            **notify,
                        }
                    )
                    sent = consumer.event("notifications_sent")
                    received = source.event("notifications_received")
                    if (
                        sent["id"] != name
                        or received["id"] != name
                        or (
                            sent["sent"] != received["received"]
                            or sent["sent"] != design["notifications"]
                        )
                    ):
                        raise RuntimeError("Notification oracle failed")
                    cleanup = [consumer.stop(), source.stop()]
                    warm = [s["native_read_ms"] for s in samples[1 + design["warmup"] :]]
                    row = {
                        "id": name,
                        "direction": direction,
                        "pair": pair,
                        "variant": variant,
                        "source_ready": source.ready,
                        "consumer_ready": consumer.ready,
                        "samples": samples,
                        "cold_ms": samples[0]["native_read_ms"],
                        "warm_median_ms": statistics.median(warm),
                        "warm_p99_ms": sorted(warm)[math.ceil(len(warm) * 0.99) - 1],
                        "notification_sent": sent,
                        "notification_received": received,
                        "cleanup": cleanup,
                    }
                    cells.append(row)
                    rows[variant] = row
                    (out / (name + ".json")).write_text(json.dumps(row, indent=2) + "\n")
                    print(
                        json.dumps(
                            {k: row[k] for k in ("id", "cold_ms", "warm_median_ms", "warm_p99_ms")}
                        ),
                        flush=True,
                    )
                pairs.append(
                    {
                        "direction": direction,
                        "pair": pair,
                        "cold_ratio": rows["candidate"]["cold_ms"] / rows["baseline"]["cold_ms"],
                        "warm_ratio": rows["candidate"]["warm_median_ms"]
                        / rows["baseline"]["warm_median_ms"],
                        "warm_p99_delta_ms": rows["candidate"]["warm_p99_ms"]
                        - rows["baseline"]["warm_p99_ms"],
                    }
                )
        for direction in (0, 1):
            rows = [p for p in pairs if p["direction"] == direction]
            summary = {"direction": direction}
            for label in ("cold", "warm"):
                values = [p[label + "_ratio"] for p in rows]
                summary[label + "_ratio"] = geometric(values)
                summary[label + "_95ci"] = interval(
                    values,
                    design["bootstrap_seed"] + direction + int(label == "warm") * 2,
                    design["bootstrap_draws"],
                )
            summary["max_warm_p99_delta_ms"] = max(p["warm_p99_delta_ms"] for p in rows)
            summary["pass"] = (
                summary["cold_ratio"] <= guards["cold_ratio"]
                and summary["cold_95ci"][1] < guards["cold_ci_upper"]
                and summary["warm_ratio"] <= guards["warm_ratio"]
                and summary["warm_95ci"][1] <= guards["warm_ci_upper"]
                and summary["max_warm_p99_delta_ms"] <= guards["warm_p99_delta_ms"]
            )
            summaries.append(summary)
        for host in contract["hosts"]:
            after = snapshot(contract, host["alias"])
            if after["gpu"] or after["libraries"] != contract["libraries"]:
                raise RuntimeError("Postflight changed runtime or retained GPU process")
            (out / (host["alias"] + "-postflight.json")).write_text(
                json.dumps(after, indent=2) + "\n"
            )
        result.update(
            state="VALID_COMPONENT_PASS"
            if all(s["pass"] for s in summaries)
            else "VALID_COMPONENT_FAIL",
            cells=len(cells),
            exact_read_oracles=len(cells) * total,
            exact_notification_oracles=len(cells) * design["notifications"],
            pairs=pairs,
            summary=summaries,
            production_qualified=False,
        )
    except BaseException as error:
        result.update(
            state="INVALID_FAIL_STOP" if owners else "INVALID_PRELAUNCH",
            error=repr(error),
            cells=len(cells),
            cell_process_started=bool(owners),
            held_owners=[
                {"name": p.name, "host": p.host, "ready": p.ready} for p in owners if not p.stopped
            ],
        )
        raise
    finally:
        (out / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("Optimized Python is not allowed")
    contract = json.loads(args.contract.read_text())
    if digest(Path(__file__)) != contract["input_hashes"]["controller"]:
        parser.error("Controller differs from frozen contract")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "CONTRACT.json").write_bytes(args.contract.read_bytes())
    run(contract, args.output)


if __name__ == "__main__":
    main()
