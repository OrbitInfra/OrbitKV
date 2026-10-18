"""Hold a frozen experiment's host leases through actual child and GPU drain."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import queue
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path


def write(path: Path, value: dict) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def snapshot() -> dict:
    result = {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    for key, arguments in {
        "gpu": [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        "compute": [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
    }.items():
        completed = subprocess.run(arguments, text=True, capture_output=True, timeout=15)
        if completed.returncode:
            raise RuntimeError((arguments, completed.returncode, completed.stderr))
        result[key] = completed.stdout
    result["nics"] = {}
    for name in ("mlx5_1", "mlx5_2", "mlx5_3", "mlx5_4"):
        directory = Path("/sys/class/infiniband") / name / "ports/1/counters"
        result["nics"][name] = {
            path.name: int(path.read_text()) for path in directory.iterdir() if path.is_file()
        }
    return result


def gpu_quiet(value: dict) -> bool:
    return not value["compute"].strip() and all(
        int(line.rsplit(",", 1)[1]) == 0 for line in value["gpu"].splitlines()
    )


def descendants() -> list[dict]:
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            pid = int(path.parent.name)
            table[pid] = {
                "pid": pid,
                "ppid": int(fields[1]),
                "state": fields[0],
                "start_ticks": int(fields[19]),
            }
        except (FileNotFoundError, ProcessLookupError):
            continue
    owned = {os.getpid()}
    while True:
        larger = owned | {pid for pid, value in table.items() if value["ppid"] in owned}
        if larger == owned:
            return [table[pid] for pid in sorted(owned - {os.getpid()})]
        owned = larger


class HostOwner:
    """Own leased host resources, subprocess pipes and their literal wait results."""

    def __init__(self, contract: dict, host: str, output: Path):
        self.contract, self.host, self.output = contract, host, output
        self.children = {}
        self.history = []
        self.leases = []
        self.uncertain = False
        self.work_admitted = False

    def runtime_checks(self) -> tuple[dict, dict]:
        declaration = self.contract["runtime_inputs"][self.host]
        contents = {}
        for name in ("helper", "guard"):
            path = Path(declaration[name])
            if not path.is_absolute():
                raise RuntimeError("Runtime checks require absolute frozen inputs")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise RuntimeError("Runtime checks require regular files")
                content = source.read(1024 * 1024 + 1)
            if (
                len(content) > 1024 * 1024
                or hashlib.sha256(content).hexdigest()
                != self.contract["runtime_hashes"][self.host][str(path)]
            ):
                raise RuntimeError(f"Frozen runtime check changed: {path}")
            contents[name] = content
        checks = {"__name__": "_orbitkv_rdma_runtime", "__file__": declaration["helper"]}
        exec(compile(contents["helper"], declaration["helper"], "exec"), checks)
        return checks, json.loads(contents["guard"])

    def validate_runtime(self, deadline: float | None = None) -> dict:
        if deadline is None:
            deadline = time.monotonic() + 60
        checks, guard = self.runtime_checks()
        if [row["path"] for row in guard["locks"]] != self.contract["resource_locks"]:
            raise RuntimeError("Runtime guard and owner resource leases differ")
        checks["check_inputs"](
            guard,
            locks=[lease.fileno() for lease in self.leases],
            deadline=deadline,
        )
        working_directory = guard["working_directory"]
        cwd = Path(working_directory["path"])
        metadata = cwd.lstat()
        if (
            not cwd.is_absolute()
            or cwd.resolve(strict=True) != cwd
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino)
            != (working_directory["device"], working_directory["inode"])
            or not any(
                declaration["path"] == working_directory["manifest"]
                for declaration in guard["runtime_manifests"]
            )
            or checks["read_json"](Path(working_directory["manifest"]), deadline)["root"]
            != str(cwd)
        ):
            raise RuntimeError("Working directory is not a frozen canonical runtime root")
        domains = guard["channel_domains"]
        if not isinstance(domains, list) or len(domains) > 128:
            raise RuntimeError("Private channel domain inventory exceeds its bound")
        cells = {cell["id"] for cell in self.contract.get("cells", []) if cell["kind"] == "manager"}
        if len(domains) != len(cells) or {domain["cell"] for domain in domains} != cells:
            raise RuntimeError("Private channel domains differ from the frozen Manager cells")
        for domain in domains:
            checks["check_runtime_domain"](domain, deadline)
            path = Path(domain["config_path"])
            prefix = domain["prefix"]
            if (
                not path.is_absolute()
                or path.resolve(strict=True) != path
                or str(path) not in guard["input_sha256"]
                or not isinstance(prefix, str)
                or not prefix
            ):
                raise RuntimeError("Private channel configuration is not frozen")
            expected = (
                f"[global]\nroot-path = {json.dumps(domain['root'], ensure_ascii=False)}\n"
                f"prefix = {json.dumps(prefix, ensure_ascii=False)}\n"
                '[global.node]\ndirectory = "nodes"\n'
                '[global.service]\ndirectory = "services"\n'
            ).encode()
            if checks["read_bytes"](path, 65536, deadline) != expected:
                raise RuntimeError("Channel configuration differs from its declared private domain")
        return guard

    def acquire(self) -> None:
        if (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            != self.contract["hosts"][self.host]["boot_id"]
        ):
            raise RuntimeError("Host boot changed")
        for name in self.contract["resource_locks"]:
            path = Path(name)
            if path.with_suffix(".quarantine.json").exists():
                raise RuntimeError(f"Resource quarantined: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a")
            self.leases.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.validate_runtime()
        before = snapshot()
        expected_uuids = self.contract["hosts"][self.host]["gpu_uuids"]
        actual_uuids = {
            row.split(",")[0].strip(): row.split(",")[1].strip()
            for row in before["gpu"].splitlines()
        }
        if actual_uuids != expected_uuids:
            raise RuntimeError("Physical GPU inventory changed")
        if not gpu_quiet(before):
            raise RuntimeError("Host has a GPU workload")
        write(self.output / "BEFORE.json", before)

    def quarantine(self, error: str) -> None:
        self.uncertain = True
        value = {
            "host": self.host,
            "owner_pid": os.getpid(),
            "error": error,
            "run": str(self.output),
            "children": list(self.children),
        }
        paths = [
            self.output / "OWNERSHIP-UNKNOWN.json",
            *[Path(handle.name).with_suffix(".quarantine.json") for handle in self.leases],
        ]
        for path in paths:
            try:
                if not path.exists():
                    write(path, value)
            except OSError as error:
                print(f"Cannot persist quarantine {path}: {error}", file=sys.stderr, flush=True)

    def event(self, value: dict) -> dict:
        self.history.append({"wall_ns": time.time_ns(), **value})
        path = self.output / "EVENTS.json"
        temporary = self.output / "EVENTS.next"
        temporary.write_text(json.dumps(self.history, indent=2) + "\n")
        temporary.replace(path)
        return value

    def launch(self, message: dict) -> dict:
        if self.uncertain:
            raise RuntimeError("Quarantined ownership blocks new admission")
        if (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            != self.contract["hosts"][self.host]["boot_id"]
        ):
            raise RuntimeError("Host boot changed before launch")
        name, cell = message["name"], message["cell"]
        planned = self.contract["launches"][self.host][cell][name]
        if name in self.children or message["argv"] != planned["argv"]:
            raise ValueError("Only the exact frozen command may launch")
        if message.get("environment", {}) != planned["environment"]:
            raise ValueError("Frozen child environment differs")
        guard = self.validate_runtime()
        if planned["cwd"] != guard["working_directory"]["path"]:
            raise RuntimeError("Child working directory differs from its frozen runtime root")
        for path, expected in self.contract["file_stamps"][self.host].items():
            info = Path(path).stat()
            if [
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ] != expected:
                raise RuntimeError(f"Frozen file changed: {path}")
        label = cell + "-" + name
        stdout = (self.output / (label + ".stdout.log")).open("x")
        stderr = (self.output / (label + ".stderr.log")).open("x")
        environment = dict(planned["environment"])
        if "MC_FORCE_TCP" in environment:
            raise RuntimeError("RDMA requires MC_FORCE_TCP to be absent")
        if planned["kind"] in ("manager", "fixture"):
            domains = [domain for domain in guard["channel_domains"] if domain["cell"] == cell]
            if (
                len(domains) != 1
                or environment.get("ORBITKV_CHANNEL_CONFIG") != domains[0]["config_path"]
                or domains[0]["config_path"] not in guard["input_sha256"]
            ):
                raise RuntimeError("Child channel configuration differs from its frozen domain")
        argv = list(planned["argv"])
        if planned["kind"] == "gpu":
            for lease in self.leases:
                argv += ["--resource-lock-fd", str(lease.fileno())]
        protocol = planned["kind"] == "gpu"
        try:
            process = subprocess.Popen(
                argv,
                cwd=planned["cwd"],
                env=environment,
                stdin=subprocess.PIPE if protocol else subprocess.DEVNULL,
                stdout=subprocess.PIPE if protocol else stdout,
                stderr=stderr,
                start_new_session=True,
                pass_fds=tuple(lease.fileno() for lease in self.leases),
                text=True,
                bufsize=1,
            )
        except BaseException:
            stdout.close()
            stderr.close()
            raise
        item = {
            "process": process,
            "stdout": stdout,
            "stderr": stderr,
            "cell": cell,
            "kind": planned["kind"],
            "queue": queue.Queue(maxsize=64),
            "native_stopped": False,
        }
        self.children[name] = item
        self.work_admitted = True
        item["start_ticks"] = int(
            (Path("/proc") / str(process.pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
        )
        if protocol:

            def reader():
                try:
                    while True:
                        line = process.stdout.readline(1024 * 1024 + 1)
                        if not line:
                            break
                        if len(line) > 1024 * 1024:
                            raise RuntimeError("Oversized native protocol output")
                        stdout.write(line)
                        stdout.flush()
                        try:
                            value = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(value, dict) and value.get("tent_gpu_endpoint"):
                            item["queue"].put(value, timeout=5)
                except BaseException as error:
                    item["queue"].put({"event": "protocol_error", "error": str(error)}, timeout=5)
                finally:
                    item["queue"].put({"event": "eof"}, timeout=5)

            item["thread"] = threading.Thread(target=reader, daemon=True)
            item["thread"].start()
        return self.event(
            {
                "event": "launched",
                "host": self.host,
                "cell": cell,
                "name": name,
                "pid": process.pid,
                "argv": argv,
                "kind": planned["kind"],
            }
        )

    def call(self, message: dict) -> dict:
        if self.uncertain:
            raise RuntimeError("Quarantined native ownership cannot accept more payload commands")
        item = self.children[message["name"]]
        if item["kind"] != "gpu":
            raise ValueError("Only GPU endpoints expose the framed protocol")
        request = message.get("request")
        if request is not None:
            item["process"].stdin.write(json.dumps(request) + "\n")
            item["process"].stdin.flush()
        try:
            value = item["queue"].get(timeout=90)
        except queue.Empty:
            self.quarantine("Native command did not establish its terminal response")
            raise
        expected = (
            "ready"
            if request is None
            else "stopped"
            if request["operation"] == "stop"
            else "result"
        )
        if value.get("event") != expected or (
            expected == "result" and value.get("id") != request["id"]
        ):
            self.quarantine(f"Unexpected native frame: {expected}, {value}")
            raise RuntimeError(("Unexpected native frame", expected, value))
        if expected == "stopped":
            if not (
                value.get("native_stopped") is True
                and value.get("registered_regions") == 0
                and value.get("gpu_allocations") == 0
            ):
                raise RuntimeError("No complete native stop receipt")
            item["native_stopped"] = True
        return value

    def process_status(self, message: dict) -> dict:
        item = self.children[message["name"]]
        process = item["process"]
        code = process.poll()
        result = {
            "pid": process.pid,
            "host": self.host,
            "boot_id": self.contract["hosts"][self.host]["boot_id"],
            "start_ticks": item["start_ticks"],
            "cell": item["cell"],
            "name": message["name"],
            "exit_code": code,
            "actual_wait_reap": code is not None,
            "listen_ports": [],
        }
        if code is not None:
            return result
        directory = Path("/proc") / str(process.pid)
        inodes = set()
        try:
            for path in (directory / "fd").iterdir():
                try:
                    target = path.readlink().as_posix()
                except FileNotFoundError:
                    continue
                if target.startswith("socket:["):
                    inodes.add(target[8:-1])
            for name in ("tcp", "tcp6"):
                for row in (directory / "net" / name).read_text().splitlines()[1:]:
                    fields = row.split()
                    if fields[3] == "0A" and fields[9] in inodes:
                        result["listen_ports"].append(int(fields[1].rsplit(":", 1)[1], 16))
        except FileNotFoundError:
            if process.poll() is None:
                raise
        # A child can exit while procfs is inspected. poll() records its real wait status.
        code = process.poll()
        result.update(exit_code=code, actual_wait_reap=code is not None)
        if code is not None:
            result["listen_ports"] = []
        return result

    def admit_runtime(self, message: dict) -> dict:
        if self.uncertain:
            raise RuntimeError("Unknown ownership blocks runtime admission")
        deadline = time.monotonic() + 60
        cell = message["cell"]
        self.validate_runtime(deadline)
        checks, guard = self.runtime_checks()
        frozen = {
            str(Path(path).resolve(strict=True)): sha for path, sha in guard["input_sha256"].items()
        }
        for declaration in guard["runtime_manifests"]:
            raw = checks["read_bytes"](
                Path(declaration["path"]), checks["MAX_RUNTIME_MANIFEST_BYTES"], deadline
            )
            if hashlib.sha256(raw).hexdigest() != declaration["sha256"]:
                raise RuntimeError("Runtime mapping manifest changed")
            for row in json.loads(raw)["files"]:
                if row["realpath"] in frozen and frozen[row["realpath"]] != row["sha256"]:
                    raise RuntimeError("Conflicting frozen runtime mapping")
                frozen[row["realpath"]] = row["sha256"]
        result = {"cell": cell, "host": self.host, "processes": {}}
        for name in ("manager", "fixture"):
            item = self.children[name]
            process = item["process"]
            if item["cell"] != cell or process.poll() is not None:
                raise RuntimeError("Runtime admission requires this cell's live owners")
            planned = self.contract["launches"][self.host][cell][name]
            expected = planned["runtime"]
            if Path(f"/proc/{process.pid}/exe").resolve(strict=True) != Path(
                expected["executable"]
            ):
                raise RuntimeError("Actual child executable differs from the frozen program")
            cwd = Path(f"/proc/{process.pid}/cwd")
            working_directory = guard["working_directory"]
            metadata = cwd.stat()
            if (
                planned["cwd"] != working_directory["path"]
                or str(cwd.resolve(strict=True)) != planned["cwd"]
                or (metadata.st_dev, metadata.st_ino)
                != (working_directory["device"], working_directory["inode"])
            ):
                raise RuntimeError("Actual child working directory differs from the frozen root")
            environment = dict(planned["environment"])
            observer = self.contract["observer_library"][self.host]
            if type(expected["observer_enabled"]) is not bool:
                raise RuntimeError("Observer mode must be explicit")
            if expected["observer_enabled"]:
                environment.update(LD_PRELOAD=observer, TENT_NATIVE_TRACE_PID=str(process.pid))
            raw_environment = checks["read_bytes"](
                Path(f"/proc/{process.pid}/environ"), 1024 * 1024, deadline
            )
            entries = [
                entry.decode().split("=", 1) for entry in raw_environment.split(b"\0") if entry
            ]
            if len(dict(entries)) != len(entries) or dict(entries) != environment:
                raise RuntimeError("Actual child environment differs from the frozen inputs")
            record = checks["capture_process_runtime"](process.pid, item["start_ticks"], deadline)
            record["cwd"] = planned["cwd"]
            paths = {row["path"] for row in record["files"]}
            if not set(expected["required_mapped_files"]) <= paths:
                raise RuntimeError("Actual child lacks its required frozen mappings")
            if (observer in paths) is not expected["observer_enabled"]:
                raise RuntimeError("Actual observer mapping differs from its frozen mode")
            for row in record["files"]:
                if frozen.get(row["realpath"]) != row["sha256"]:
                    raise RuntimeError(f"Actual runtime mapping is not frozen: {row['path']}")
            result["processes"][name] = record
        manager = self.children["manager"]["process"]
        text = ""
        for stream in ("stdout", "stderr"):
            path = self.output / f"{cell}-manager.{stream}.log"
            text += checks["read_bytes"](path, checks["MAX_LOG_BYTES"], deadline).decode() + "\n"
        checks["validate_python_origin"](
            text, self.contract["python_origins"][self.host], manager.pid, deadline
        )
        for record in result["processes"].values():
            cwd = Path(f"/proc/{record['owner']['pid']}/cwd")
            metadata = cwd.stat()
            if str(cwd.resolve(strict=True)) != record["cwd"] or (
                metadata.st_dev,
                metadata.st_ino,
            ) != (working_directory["device"], working_directory["inode"]):
                raise RuntimeError("Actual child working directory changed during inspection")
        if any(item["process"].poll() is not None for item in self.children.values()):
            raise RuntimeError("Owned child exited during runtime admission")
        path = self.output / f"{cell}-runtime-admission.json"
        checks["write"](path, result, deadline)
        return self.event(
            {
                "event": "runtime-admitted",
                "host": self.host,
                "cell": cell,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "owners": {name: record["owner"] for name, record in result["processes"].items()},
                "physical_termination_claimed": False,
            }
        )

    def stop(self, message: dict) -> dict:
        name = message["name"]
        item = self.children[name]
        process = item["process"]
        was_running = process.poll() is None
        if item["kind"] == "gpu":
            if not item["native_stopped"]:
                raise RuntimeError("Cannot stop a GPU endpoint before its native drain receipt")
        elif message.get("idle_proven") is not True:
            raise ValueError("Explicit workload idle proof required before service stop")
        elif was_running:
            process.send_signal(signal.SIGTERM)
        try:
            code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.quarantine(f"Child {name} did not establish termination")
            raise
        if item["kind"] == "gpu":
            item["thread"].join(timeout=5)
            if item["thread"].is_alive():
                self.quarantine(f"Child {name} protocol reader is still active")
                raise RuntimeError("Protocol reader not drained")
            process.stdin.close()
            process.stdout.close()
        item["stdout"].close()
        item["stderr"].close()
        del self.children[name]
        return self.event(
            {
                "event": "stopped",
                "host": self.host,
                "cell": item["cell"],
                "name": name,
                "pid": process.pid,
                "exit_code": code,
                "was_running": was_running,
                "actual_wait_reap": True,
                "native_stopped": item["native_stopped"],
                "forced_cleanup": False,
                "boot_id": self.contract["hosts"][self.host]["boot_id"],
                "start_ticks": item["start_ticks"],
            }
        )

    def finish(self) -> dict:
        if self.children or self.uncertain:
            raise RuntimeError("Unknown/live ownership blocks lease release")
        deadline = time.monotonic() + 60
        quiet_until = time.monotonic() + 30
        adopted = []
        while True:
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not pid:
                    break
                adopted.append({"pid": pid, "exit_code": os.waitstatus_to_exitcode(status)})
            try:
                after = snapshot()
                remaining = descendants()
            except Exception as error:
                self.quarantine(f"Cannot establish physical postflight: {error}")
                raise
            after.update(descendants=remaining, adopted_reaped=adopted)
            sockets = [
                path
                for path in self.contract.get("owned_sockets", {}).get(self.host, [])
                if Path(path).exists()
            ]
            after["remaining_sockets"] = sockets
            if gpu_quiet(after) and not remaining and not sockets:
                try:
                    self.validate_runtime(deadline)
                except Exception as error:
                    self.quarantine(f"Cannot validate final runtime inputs: {error}")
                    raise
                write(self.output / "AFTER.json", after)
                return after
            if time.monotonic() >= quiet_until:
                self.quarantine("GPU allocations outlive declared child termination")
                raise RuntimeError("GPU not drained")
            time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        raise RuntimeError("Optimized guards forbidden")
    input_index = args.contract.parent / "INPUTS.json"
    input_bytes = input_index.read_bytes()
    receipt = json.loads((args.contract.parent / "LAUNCH-ACCEPTED.json").read_text())
    if (
        receipt.get("accepted") is not True
        or receipt.get("inputs_sha256") != hashlib.sha256(input_bytes).hexdigest()
    ):
        raise RuntimeError("Exact inputs lack independent launch acceptance")
    inputs = json.loads(input_bytes)
    contract_bytes = args.contract.read_bytes()
    if hashlib.sha256(contract_bytes).hexdigest() != inputs.get(args.contract.name):
        raise RuntimeError("Frozen contract changed")
    for name, digest in inputs.items():
        content = (
            contract_bytes
            if name == args.contract.name
            else (args.contract.parent / name).read_bytes()
        )
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"Frozen input changed: {name}")
    contract = json.loads(contract_bytes)
    resources = contract["resource_locks"]
    if (
        not isinstance(resources, list)
        or not 1 <= len(resources) <= 64
        or any(not isinstance(path, str) or not Path(path).is_absolute() for path in resources)
        or len(set(resources)) != len(resources)
    ):
        raise RuntimeError("RDMA requires a bounded nonempty inventory of resource leases")
    for path, digest in contract["runtime_hashes"][args.host].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen runtime changed: {path}")
    args.output.mkdir(exist_ok=False)
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0):
        raise RuntimeError("Cannot establish child subreaper")
    owner = HostOwner(contract, args.host, args.output)
    finished = False
    try:
        owner.acquire()
        print(
            json.dumps({"rdma_owner": True, "ready": True, "host": args.host, "pid": os.getpid()}),
            flush=True,
        )
        while True:
            line = sys.stdin.readline(16385)
            if not line or len(line) > 16384:
                owner.quarantine("Controller EOF/oversized frame retains leases and children")
                break
            try:
                request = json.loads(line)
            except (ValueError, TypeError) as error:
                owner.quarantine(f"Invalid controller frame: {error}")
                break
            try:
                operation = request["operation"]
                if operation == "launch":
                    result = owner.launch(request)
                elif operation == "call":
                    result = owner.call(request)
                elif operation == "stop":
                    result = owner.stop(request)
                elif operation == "snapshot":
                    result = snapshot()
                elif operation == "process-status":
                    result = owner.process_status(request)
                elif operation == "admit-runtime":
                    result = owner.admit_runtime(request)
                elif operation == "finish":
                    result = owner.finish()
                    finished = True
                else:
                    raise ValueError("Unknown owner operation")
                response = {"ok": True, "id": request["id"], "result": result}
            except Exception as error:
                response = {
                    "ok": False,
                    "id": request["id"],
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "ownership_uncertain": owner.uncertain,
                }
            print(json.dumps({"rdma_owner": True, **response}), flush=True)
            if finished:
                break
    finally:
        if not finished and (owner.children or owner.work_admitted):
            owner.quarantine("Owner did not establish complete native child drain")
            while True:
                signal.pause()
        if not finished and owner.uncertain:
            while True:
                signal.pause()
        for lease in owner.leases:
            lease.close()


if __name__ == "__main__":
    main()
