#!/usr/bin/env python3
"""Required CPU Manager/process-crash gate. Real OrbiTE file I/O, no GPU claim.

Use a test-hooks Manager and the host-registration shim. The filesystem root
must be a dedicated, bounded ext4/tmpfs mount: the disk-full control fills it.
Logs, results and recovery measurements go to --artifacts, outside the checkout.
"""

import argparse
import errno
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def frame(value):
    data = json.dumps(value, separators=(",", ":")).encode()
    return struct.pack("<II", len(data), crc32c(data)) + data


def frames(path):
    data = path.read_bytes()
    values = []
    while data:
        size, crc = struct.unpack("<II", data[:8])
        body, data = data[8 : 8 + size], data[8 + size :]
        assert crc32c(body) == crc
        values.append(json.loads(body))
    return values


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Manager:
    def __init__(self, gate, path, *, capacity=4096, persistent=True, pauses=()):
        self.gate, self.sequence = gate, 0
        self.root = Path(tempfile.mkdtemp(prefix="okv-ssd-"))
        self.prefix = f"ssd_{uuid.uuid4().hex}_"
        self.control = self.root / "control"
        self.control.mkdir()
        domain = self.root / "domain"
        domain.mkdir()
        (domain / "nodes").mkdir()
        (domain / "services").mkdir()
        config = self.root / "channel.toml"
        config.write_text(f'[global]\nroot-path="{domain}"\nprefix="{self.prefix}"\n')
        for pause in pauses:
            self.pause(pause)
        self.log_path = gate.artifacts / f"{len(gate.logs):03d}-manager.log"
        gate.logs.append(str(self.log_path))
        self.log = self.log_path.open("wb")
        command = [
            str(gate.binary),
            "--ssd-cache-path",
            str(path),
            "--ssd-cache-capacity",
            str(capacity),
            "--ssd-backend",
            "uring",
            "--pool-size",
            "16mb",
            "--disable-numa-affinity",
            "--transfer-transport",
            "tcp",
            "--addr",
            f"127.0.0.1:{port()}",
            "--http-addr",
            f"127.0.0.1:{port()}",
            "--bootstrap-socket",
            str(self.root / "m.sock"),
        ]
        if persistent:
            command.append("--ssd-persistent")
        self.process = subprocess.Popen(
            command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "ORBITKV_TEST_SSD_PROCESS": str(self.control),
                "ORBITKV_TEST_FAULTS": str(self.control),
                "ORBITKV_CHANNEL_CONFIG": str(config),
            },
        )
        gate.children.append(self)
        self.wait(lambda: (self.control / "ready").exists())
        self.wait(lambda: "Standalone Cache Manager ready" in self.log_path.read_text())
        if "ssd_before_verification" not in pauses:
            self.recovered()

    def wait(self, condition, timeout=90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert self.process.poll() is None, self.log_path.read_text()
            result = condition()
            if result:
                return result
            time.sleep(0.01)
        raise AssertionError(f"Manager gate timed out; see {self.log_path}")

    def pause(self, name):
        (self.control / f"{name}.pause").touch()

    def reached(self, name):
        self.wait(lambda: (self.control / f"{name}.reached").exists())

    def release(self, name):
        (self.control / f"{name}.pause").unlink()

    def send(self, operation, **kwargs):
        self.sequence += 1
        temporary = self.control / "command.tmp"
        temporary.write_text(
            json.dumps(dict(sequence=self.sequence, operation=operation, **kwargs))
        )
        temporary.replace(self.control / "command.json")

    def reply(self):
        def receive():
            path = self.control / "reply.json"
            if path.exists():
                value = json.loads(path.read_text())
                if value["sequence"] == self.sequence:
                    return value
            return None

        result = self.wait(receive)
        status = Path(f"/proc/{self.process.pid}/status").read_text().splitlines()
        result["state"]["process_peak_rss_kib"] = int(
            next(line for line in status if line.startswith("VmHWM:")).split()[1]
        )
        self.gate.observations.append(result["state"])
        return result

    def call(self, operation, **kwargs):
        self.send(operation, **kwargs)
        return self.reply()

    def recovered(self):
        return self.wait(
            lambda: (
                (r := self.call("report"))["state"]["recovery"] is None
                or r["state"]["recovery"]["finished"]
            )
        )

    def read(self, id=1, *, present=True, encoded=False, namespace=0):
        result = self.call("read", id=id, namespace=namespace)
        assert result["present"] == present, result
        if present:
            assert [s["bytes"] for s in result["slots"]] == [
                [id] * 137,
                [(id + 1) % 256] * 631,
            ]
            assert all(s["encoded"] == encoded for s in result["slots"])
        return result

    def close(self, kill=False):
        if self.process.poll() is None:
            if kill:
                self.process.kill()
            else:
                self.call("stop")
                self.process.terminate()
            code = self.process.wait(timeout=30)
            assert code == (-signal.SIGKILL if kill else 0), self.log_path.read_text()
        self.log.close()
        # Only this child's private channel objects; never clean a shared domain.
        for path in Path("/dev/shm").glob(self.prefix + "*"):
            path.unlink()
        shutil.rmtree(self.root)
        self.gate.children.remove(self)


class Gate:
    def __init__(self, args):
        self.binary, self.root, self.artifacts = (
            args.binary.resolve(),
            args.root.resolve(),
            args.artifacts.resolve(),
        )
        self.logs, self.observations, self.results, self.children = [], [], [], []
        self.artifacts.mkdir(parents=True, exist_ok=False)

    def case(self, name, run):
        directory = self.root / name
        directory.mkdir()
        started = time.monotonic()
        try:
            run(directory / "data")
            self.results.append(
                {"case": name, "status": "pass", "seconds": time.monotonic() - started}
            )
            print(f"PASS {name}", flush=True)
        except BaseException:
            self.results.append({"case": name, "status": "fail"})
            raise
        finally:
            for child in self.children[:]:
                child.close(kill=True)
            (self.artifacts / "results.json").write_text(
                json.dumps(
                    {
                        "claim": "CPU process failure only; not power-loss or GPU evidence",
                        "cases": self.results,
                        "recovery": self.observations,
                        "logs": self.logs,
                    },
                    indent=2,
                )
            )

    def manager(self, path, **kwargs):
        return Manager(self, path, **kwargs)

    def crash(self, path, point, encoded):
        child = self.manager(path, pauses=[point])
        child.send("write", id=1, encoded=encoded)
        child.reached(point)
        child.close(kill=True)
        restarted = self.manager(path)
        result = restarted.call("read", id=1)
        if point == "ssd_published":
            assert result["present"], result
        if result["present"]:
            restarted.read(encoded=encoded)
        else:
            assert sum(result["state"]["recovery"]["drops"].values()) > 0, result
        restarted.close()

    def clean(self, path, persistent=True):
        child = self.manager(path)
        assert child.call("write", id=1)["present"]
        assert child.call("write", id=2, encoded=True)["present"]
        child.close()
        child = self.manager(path, persistent=persistent)
        child.read(present=persistent)
        child.read(2, present=persistent, encoded=True)
        child.read(namespace=1, present=False)
        child.close()

    def verify(self, path):
        child = self.manager(path)
        assert child.call("write", id=1)["present"]
        child.close()
        child = self.manager(path, pauses=["ssd_before_verification"])
        child.reached("ssd_before_verification")
        child.read(present=False)
        child.release("ssd_before_verification")
        child.recovered()
        child.read()
        child.close()

    def corrupt(self, path, kind):
        child = self.manager(path)
        assert child.call("write", id=1)["present"]
        child.close()
        journal = Path(str(path) + ".index")
        if kind == "data":
            with path.open("r+b") as file:
                file.seek(17)
                file.write(b"\xff")
                os.fsync(file.fileno())
        elif kind == "torn":
            with journal.open("r+b") as file:
                file.truncate(journal.stat().st_size - 1)
        elif kind == "checksum":
            data = bytearray(journal.read_bytes())
            data[-1] ^= 1
            journal.write_bytes(data)
        else:
            records = frames(journal)
            if kind == "version":
                records[-1]["version"] += 1
            else:
                records[-1]["key"]["namespace"] = "wrong-model-namespace"
            journal.write_bytes(b"".join(frame(record) for record in records))
        child = self.manager(path)
        result = child.read(present=False)
        assert sum(result["state"]["recovery"]["drops"].values()) > 0, result
        child.close()

    def wrap(self, path, point):
        child = self.manager(path, capacity=1024)
        assert child.call("write", id=1)["present"]
        child.pause(point)
        child.send("write", id=2)
        child.reached(point)
        child.close(kill=True)
        child = self.manager(path, capacity=1024)
        child.read(present=False)
        child.read(2, present=False)
        assert child.call("write", id=3)["present"]  # Orphan capacity is reusable.
        child.read(3)
        child.close()

    def owners(self, path, receipt):
        child = self.manager(path, capacity=1024)
        assert child.call("write", id=1)["present"]
        if receipt:
            child.pause("ssd_read_receipt")
            child.call("read_start", id=1)
            child.reached("ssd_read_receipt")
        else:
            assert child.call("hold", id=1)["present"]
        report = child.call("write", id=2)
        assert not report["present"] and report["state"]["ring"]["readers"] > 0, report
        if receipt:
            child.release("ssd_read_receipt")
            assert child.call("read_finish")["slots"][0]["bytes"] == [1] * 137
        else:
            child.call("release")
        assert child.call("write", id=2)["present"]
        child.read(2)
        child.read(present=False)
        child.close()

    def shrink(self, path):
        child = self.manager(path, capacity=3072)
        for id in range(1, 4):
            assert child.call("write", id=id)["present"]
        child.close()
        child = self.manager(path, capacity=1024)
        result = child.read()
        assert result["state"]["ring"]["bytes"] <= 1024, result
        assert result["state"]["recovery"]["drops"]["capacity"] == 2
        child.read(2, present=False)
        child.read(3, present=False)
        child.close()

    def checkpoint(self, path, point):
        child = self.manager(path)
        assert child.call("write", id=1)["present"]
        child.pause("ssd_force_checkpoint")
        child.pause(point)
        child.send("write", id=2)
        child.reached(point)
        child.close(kill=True)
        child = self.manager(path)
        child.read()
        child.read(2, present=False)
        child.close()

    def disk_full(self, path):
        child = self.manager(path, pauses=["ssd_before_record"])
        child.send("write", id=1, wide=True)
        child.reached("ssd_before_record")
        filler = self.root / "disk-full-owned-filler"
        try:
            with filler.open("wb", buffering=0) as file:
                try:
                    while True:
                        file.write(bytes(4096))
                except OSError as error:
                    assert error.errno == errno.ENOSPC, error
            child.release("ssd_before_record")
            result = child.reply()
            assert not result["present"], result
            assert result["state"]["pending_writes"] == 0, result
            assert result["state"]["ring"]["readers"] == 0, result
            assert result["state"]["ring"]["writing"] == 0, result
        finally:
            filler.unlink(missing_ok=True)
        assert not child.call("write", id=2)["present"]  # Fenced after ENOSPC.
        child.close()
        child = self.manager(path)
        child.read(present=False)
        assert child.call("write", id=2)["present"]
        child.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--filesystem", required=True, choices=["ext4", "tmpfs"])
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    for name in ["ORBITE_REQUIRE_IO_URING", "ORBITKV_REQUIRE_ENGINE_SSD"]:
        assert os.environ.get(name) == "1", f"{name}=1 is required"
    assert (
        subprocess.check_output(
            ["findmnt", "-n", "-o", "FSTYPE", "-T", str(args.root)], text=True
        ).strip()
        == args.filesystem
    )
    assert os.path.ismount(args.root) and all(
        p.name == "lost+found" for p in args.root.iterdir()
    ), "use an empty, dedicated bounded mount"
    assert shutil.disk_usage(args.root).total <= 64 * 1024 * 1024, (
        "disk-full control requires a <=64 MiB mount"
    )
    assert crc32c(b"123456789") == 0xE3069283
    gate = Gate(args)
    for point in [
        "ssd_data_written",
        "ssd_data_synced",
        "ssd_mid_record",
        "ssd_record_written",
        "ssd_published",
    ]:
        for encoded in [False, True]:
            gate.case(
                f"{point}-encoded-{encoded}",
                lambda path, p=point, e=encoded: gate.crash(path, p, e),
            )
    gate.case("clean-restart", gate.clean)
    gate.case("default-truncates", lambda path: gate.clean(path, False))
    gate.case("verify-before-first-hit", gate.verify)
    for kind in ["data", "torn", "checksum", "version", "namespace"]:
        gate.case(f"corrupt-{kind}", lambda path, k=kind: gate.corrupt(path, k))
    for point in ["ssd_reserved", "ssd_data_written"]:
        gate.case(f"wrap-{point}", lambda path, p=point: gate.wrap(path, p))
    gate.case("reader-protects-reuse", lambda path: gate.owners(path, False))
    gate.case("receipt-protects-reuse", lambda path: gate.owners(path, True))
    gate.case("capacity-shrink", gate.shrink)
    for point in ["ssd_checkpoint_synced", "ssd_checkpoint_replaced"]:
        gate.case(point, lambda path, p=point: gate.checkpoint(path, p))
    gate.case("disk-full", gate.disk_full)
    print(f"{len(gate.results)} required cases passed ({args.filesystem})")


if __name__ == "__main__":
    main()
