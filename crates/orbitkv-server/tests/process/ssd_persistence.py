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
    def __init__(
        self,
        gate,
        path,
        *,
        capacity=4096,
        persistent=True,
        pauses=(),
        reject_with=None,
        shared_namespace=None,
        root_capacity=None,
        defer_start=False,
        shards=1,
        watermarks=None,
        address=None,
        cluster_args=(),
    ):
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
        self.http = f"127.0.0.1:{port()}"
        self.address = address or f"127.0.0.1:{port()}"
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
            self.address,
            "--http-addr",
            self.http,
            "--bootstrap-socket",
            str(self.root / "m.sock"),
            *cluster_args,
        ]
        if shared_namespace is not None:
            command.extend(
                ["--ssd-shared-namespace", f"orbitkv:v2:{shared_namespace:064x}"]
            )
        if root_capacity is not None:
            command.extend(["--ssd-shared-root-capacity", str(root_capacity)])
        if shards != 1:
            command.extend(["--ssd-cache-shards", str(shards)])
        if watermarks is not None:
            low, high = watermarks
            command.extend(
                [
                    "--ssd-low-watermark-percent",
                    str(low),
                    "--ssd-high-watermark-percent",
                    str(high),
                ]
            )
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
        self.start_message = (
            "Cache Manager control listening"
            if cluster_args
            else "Standalone Cache Manager ready"
        )
        self.wait_for_recovery = not {
            "ssd_before_verification",
            "ssd_watermark_invalidated",
        }.intersection(pauses)
        if reject_with is not None:
            self.rejected(reject_with)
        elif not defer_start:
            self.ready()

    def rejected(self, reason):
        assert self.process.wait(timeout=30) != 0
        log = self.log_path.read_text()
        assert reason in log, log

    def ready(self):
        self.wait(lambda: (self.control / "ready").exists())
        self.wait(lambda: self.start_message in self.log_path.read_text())
        if self.wait_for_recovery:
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

    def wait_for_root_lock(self, root):
        inode = root.stat().st_ino

        def blocked():
            for line in Path("/proc/locks").read_text().splitlines():
                fields = line.split()
                if (
                    len(fields) == 9
                    and fields[1:3] == ["->", "FLOCK"]
                    and fields[5] == str(self.process.pid)
                    and fields[6].endswith(f":{inode}")
                ):
                    return True
            return False

        self.wait(blocked)

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

    def read(
        self,
        id=1,
        *,
        present=True,
        encoded=False,
        namespace=0,
        operation="read",
        fill=None,
    ):
        result = self.call(operation, id=id, namespace=namespace)
        assert result["present"] == present, result
        if present:
            expected = id if fill is None else fill
            assert [s["bytes"] for s in result["slots"]] == [
                [expected] * 137,
                [(expected + 1) % 256] * 631,
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

    def shared_namespaces(self, root):
        root.mkdir()
        block = 17
        first, second = 1, 2
        left = self.manager(
            root, capacity=1024, shared_namespace=first, pauses=["ssd_data_written"]
        )
        right = self.manager(root, capacity=1024, shared_namespace=second)
        left.send("write", id=block, namespace=first, fill=41)
        left.reached("ssd_data_written")
        published = right.call("write", id=block, namespace=second, fill=93)
        assert published["present"], published
        right.read(block, namespace=second, fill=93)
        left.release("ssd_data_written")
        committed = left.reply()
        assert committed["present"] and committed["hash"] == published["hash"], (
            committed
        )
        left.read(block, namespace=first, fill=41)

        for manager, foreign in ((left, second), (right, first)):
            manager.read(block, namespace=foreign, present=False)
            denied = manager.call("write", id=block, namespace=foreign, fill=181)
            assert not denied["present"], denied
        # A different shard count still contends for the same first inode.
        self.manager(
            root,
            shared_namespace=first,
            shards=2,
            reject_with="SSD shard is already owned",
        ).close()
        left.close(kill=True)
        left = self.manager(root, capacity=1024, shared_namespace=first)
        left.read(block, namespace=first, fill=41)
        left.read(block, namespace=second, present=False)
        right.read(block, namespace=second, fill=93)
        left.pause("ssd_read_receipt")
        left.call("read_start", id=block, namespace=first)
        left.reached("ssd_read_receipt")
        blocked = left.call("write", id=block + 1, namespace=first)
        assert not blocked["present"] and blocked["state"]["ring"]["readers"] > 0, (
            blocked
        )
        assert right.call("write", id=block + 1, namespace=second)["present"]
        right.read(block, namespace=second, present=False)
        right.read(block + 1, namespace=second)
        left.release("ssd_read_receipt")
        finished = left.call("read_finish")["slots"]
        assert [s["bytes"] for s in finished] == [[41] * 137, [42] * 631]
        assert left.call("write", id=block + 1, namespace=first)["present"]
        left.read(block + 1, namespace=first)
        lengths = [p.stat().st_size for p in sorted(root.glob("*/shard-*.dat"))]
        assert lengths == [1024, 1024], lengths
        (self.artifacts / "shared-namespace.json").write_text(
            json.dumps(
                {
                    "block_hash": committed["hash"],
                    "partition_file_bytes": lengths,
                    "held_read_slot_bytes": [len(s["bytes"]) for s in finished],
                    "readers_while_blocked": blocked["state"]["ring"]["readers"],
                },
                indent=2,
            )
        )
        right.close()
        left.close()

    def shared_root_capacity(self, root):
        root.mkdir()
        original = {"shared_namespace": 1, "capacity": 2048, "shards": 2}
        legacy = self.manager(root, **original)
        assert legacy.call("write", id=1, namespace=1)["present"]
        self.manager(
            root,
            **original,
            root_capacity=4096,
            reject_with="SSD shard is already owned",
        ).close()
        assert not (root / ".orbitkv-ssd-capacity").exists()
        legacy.close(kill=True)

        # Adoption includes both existing shards before any new partition is created.
        self.manager(
            root,
            shared_namespace=2,
            capacity=1024,
            root_capacity=2048,
            reject_with="shared SSD root capacity exceeded",
        ).close()
        assert len(list(root.iterdir())) == 1
        winner = self.manager(
            root,
            shared_namespace=2,
            capacity=1024,
            root_capacity=3072,
            pauses=["ssd_root_admitted"],
            defer_start=True,
        )
        winner.reached("ssd_root_admitted")
        challenger = self.manager(
            root, shared_namespace=3, capacity=1024, defer_start=True
        )
        # Wait for the second process's actual lock request while the winning shard is still unsized.
        challenger.wait_for_root_lock(root)
        winner.release("ssd_root_admitted")
        winner.ready()
        challenger.rejected("shared SSD root capacity exceeded")
        challenger.close()
        assert winner.call("write", id=2, namespace=2)["present"]
        self.manager(
            root,
            shared_namespace=3,
            capacity=1024,
            root_capacity=8192,
            reject_with="shared SSD root capacity is 3072",
        ).close()
        winner.close(kill=True)

        # Restart at the limit counts the original partition once. Killing both writers
        # leaves all reservations on disk, even for a new Manager that omits the flag.
        self.manager(
            root,
            shared_namespace=3,
            capacity=1024,
            reject_with="shared SSD root capacity exceeded",
        ).close()
        self.manager(
            root,
            shared_namespace=1,
            capacity=3072,
            shards=2,
            reject_with="shared SSD root capacity exceeded",
        ).close()
        left = self.manager(root, **original)
        left.read(1, namespace=1)
        left.close()
        right = self.manager(root, shared_namespace=2, capacity=1024)
        right.read(2, namespace=2)
        right.close()
        smaller = self.manager(root, shared_namespace=1, capacity=1024, shards=2)
        smaller.close()
        interrupted = self.manager(
            root,
            shared_namespace=4,
            capacity=1024,
            pauses=["ssd_root_admitted"],
            defer_start=True,
        )
        interrupted.reached("ssd_root_admitted")
        interrupted.close(kill=True)
        third = self.manager(root, shared_namespace=3, capacity=1024)
        assert third.call("write", id=3, namespace=3)["present"]
        third.read(3, namespace=3)
        third.close()
        assert sum(p.stat().st_size for p in root.glob("*/shard-*.dat")) == 3072

    def reclaim_offline_partition(self, root):
        root.mkdir()
        donor_config = {"shared_namespace": 1, "capacity": 2048, "shards": 2}
        donor = self.manager(root, **donor_config, root_capacity=3072)
        neighbor = self.manager(root, shared_namespace=2, capacity=1024)
        assert donor.call("write", id=1, namespace=1)["present"]
        assert neighbor.call("write", id=2, namespace=2)["present"]
        namespace = f"orbitkv:v2:{1:064x}"
        directory = root / f"orbitkv-ssd-{namespace}"
        shards = sorted(directory.glob("shard-*.dat"))
        journal = directory / "shard-000000.dat.index"
        policy = root / ".orbitkv-ssd-capacity"
        initial_policy = policy.read_bytes()
        command = [
            str(self.binary),
            "reclaim-ssd-partition",
            "--root",
            str(root),
            "--namespace",
            namespace,
        ]

        def reclaim(*, released=None, denied=None):
            run = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=30
            )
            if denied is not None:
                assert run.returncode != 0 and denied in run.stderr, run
            else:
                assert run.returncode == 0, run.stderr
                assert (
                    run.stdout.strip()
                    == f"Reclaimed {released} payload bytes from {namespace}"
                ), run

        def crash_at(point, *, concurrent_admission=False):
            challenger = None
            with tempfile.TemporaryDirectory() as tmp:
                pause = Path(tmp) / (point + ".pause")
                pause.touch()
                with subprocess.Popen(
                    command, env={**os.environ, "ORBITKV_TEST_FAULTS": tmp}
                ) as child:
                    try:
                        end = time.monotonic() + 30
                        while not pause.with_suffix(".reached").exists():
                            assert child.poll() is None and time.monotonic() < end, (
                                point
                            )
                            time.sleep(0.01)
                        if concurrent_admission:
                            challenger = self.manager(
                                root,
                                shared_namespace=3,
                                capacity=2048,
                                defer_start=True,
                            )
                            challenger.wait_for_root_lock(root)
                    finally:
                        child.kill()
                    assert child.wait(timeout=10) == -signal.SIGKILL
            if challenger is not None:
                challenger.rejected("shared SSD root capacity exceeded")
                challenger.close()

        def require_no_space():
            self.manager(
                root,
                shared_namespace=3,
                capacity=2048,
                reject_with="shared SSD root capacity exceeded",
            ).close()

        reclaim(denied="SSD shard is already owned")
        donor.read(1, namespace=1)
        assert len(shards) == 2
        with shards[1].open("rb") as old_reader:
            donor.close(kill=True)
            original = [p.read_bytes() for p in [journal, *shards]]
            reclaim(denied="drained prior file users")
            assert [p.read_bytes() for p in [journal, *shards]] == original
            assert old_reader.read() == original[-1]
        require_no_space()

        # A stopped partition with a replacement inode does not authorize eviction.
        saved = root / "original-shard"
        shards[1].rename(saved)
        shards[1].write_bytes(original[-1])
        reclaim(denied="metadata_identity")
        assert [p.read_bytes() for p in [journal, *shards]] == original
        saved.replace(shards[1])
        alias = root / "journal-alias"
        os.link(journal, alias)
        reclaim(denied="singly linked")
        assert alias.read_bytes() == original[0]
        alias.unlink()

        crash_at("ssd_reclaim_journal_synced")
        assert len(frames(journal)) == 1
        assert [p.read_bytes() for p in shards] == original[1:]
        require_no_space()
        recovered = self.manager(root, **donor_config)
        recovered.read(1, namespace=1, present=False)
        assert recovered.call("write", id=1, namespace=1)["present"]
        recovered.close()

        crash_at("ssd_reclaim_shard_synced", concurrent_admission=True)
        assert len(frames(journal)) == 1
        assert [p.stat().st_size for p in shards] == [0, 1024]
        reclaim(released=1024)
        reclaim(released=0)
        assert [p.stat().st_size for p in shards] == [0, 0]
        newcomer = self.manager(root, shared_namespace=3, capacity=2048)
        assert newcomer.call("write", id=3, namespace=3)["present"]
        newcomer.read(3, namespace=3)
        neighbor.read(2, namespace=2)
        assert policy.read_bytes() == initial_policy
        newcomer.close()
        neighbor.close()

    def relinquish_live_partition(self, root):
        root.mkdir()
        donor = self.manager(
            root, shared_namespace=1, capacity=2048, root_capacity=3072
        )
        neighbor = self.manager(root, shared_namespace=2, capacity=1024)
        for key in (5, 6):
            assert donor.call("write", id=key, namespace=1)["present"]
        donor.read(5, namespace=1, operation="peer_read")  # warm one copy in DRAM
        assert neighbor.call("write", id=8, namespace=2)["present"]
        namespace = f"orbitkv:v2:{1:064x}"
        partition = root / f"orbitkv-ssd-{namespace}"
        shard, journal = (
            partition / "shard-000000.dat",
            partition / "shard-000000.dat.index",
        )
        original = (shard.read_bytes(), journal.read_bytes())
        command = [
            str(self.binary),
            "reclaim-ssd-partition",
            "--root",
            str(root),
            "--namespace",
            namespace,
            "--wait-for-owner-secs",
        ]
        donor.pause("ssd_read_receipt")
        donor.call("read_start", id=6, namespace=1)
        donor.reached("ssd_read_receipt")
        with shard.open("rb") as prior_file_user:
            with subprocess.Popen(command + ["90"]) as request:
                try:
                    donor.wait(
                        lambda: not donor.call("report")["state"]["ssd_attached"]
                    )
                    donor.read(6, namespace=1, present=False)
                    assert not donor.call("write", id=7, namespace=1)["present"]
                    donor.read(5, namespace=1, operation="peer_read")
                    neighbor.read(8, namespace=2)
                    self.manager(
                        root,
                        shared_namespace=1,
                        capacity=2048,
                        reject_with="shared SSD partition is being reclaimed",
                    ).close()
                    self.manager(
                        root,
                        shared_namespace=3,
                        capacity=2048,
                        reject_with="shared SSD root capacity exceeded",
                    ).close()
                    assert request.poll() is None
                    assert (shard.read_bytes(), journal.read_bytes()) == original
                finally:
                    request.kill()
                assert request.wait(timeout=10) == -signal.SIGKILL
            # The admitted read still returns its original bytes after detachment.
            donor.release("ssd_read_receipt")
            donor.read(6, namespace=1, operation="read_finish")
            timed_out = subprocess.run(
                command + ["1"], check=False, capture_output=True, text=True, timeout=15
            )
            assert (
                timed_out.returncode != 0
                and "drained prior file users" in timed_out.stderr
            ), timed_out
            assert not donor.call("report")["state"]["ssd_attached"]
            assert (prior_file_user.read(), journal.read_bytes()) == original

        completed = subprocess.run(
            command + ["30"], check=True, capture_output=True, text=True, timeout=40
        )
        assert (
            completed.stdout.strip() == f"Reclaimed 2048 payload bytes from {namespace}"
        )
        assert shard.stat().st_size == 0 and len(frames(journal)) == 1
        newcomer = self.manager(root, shared_namespace=3, capacity=2048)
        assert newcomer.call("write", id=9, namespace=3)["present"]
        newcomer.read(9, namespace=3)
        neighbor.read(8, namespace=2)
        donor.read(5, namespace=1, operation="peer_read")
        newcomer.close()
        neighbor.close()
        donor.close()

    def shared_partition_reopen(self, root):
        root.mkdir()
        scope = 7
        args = {"capacity": 2048, "shared_namespace": scope, "shards": 2}
        original = self.manager(root, **args)
        for block in (1, 2):
            assert original.call("write", id=block, namespace=scope)["present"]
        original.close()
        (directory,) = root.iterdir()
        data = sorted(directory.glob("shard-*.dat"))
        journal = directory / "shard-000000.dat.index"
        before = {p: p.read_bytes() for p in directory.iterdir()}
        assert [len(before[p]) for p in data] == [1024, 1024]
        self.manager(
            root,
            capacity=2048,
            shared_namespace=scope,
            reject_with="shard count change",
        ).close()
        self.manager(
            root,
            capacity=4096,
            shared_namespace=scope,
            shards=4,
            reject_with="shard count changed",
        ).close()
        for shard in data:
            self.manager(
                shard,
                persistent=False,
                reject_with="inside a managed SSD partition",
            ).close()
        alias = root / "partition-alias"
        alias.symlink_to(directory, target_is_directory=True)
        self.manager(
            alias / "nested" / "cache.dat",
            persistent=False,
            reject_with="inside a managed SSD partition",
        ).close()
        self.manager(
            alias,
            shared_namespace=scope + 1,
            reject_with="inside a managed SSD partition",
        ).close()
        assert {p: p.read_bytes() for p in directory.iterdir()} == before
        journal.unlink()
        self.manager(
            root, **args, reject_with="missing shared SSD partition header"
        ).close()
        assert all(p.read_bytes() == before[p] for p in data)
        assert not journal.exists()
        journal.write_bytes(before[journal])
        restarted = self.manager(root, **args)
        for block in (1, 2):
            restarted.read(block, namespace=scope)
        restarted.close()

    def ttl(self, path, during_verification):
        point = "ssd_before_verification" if during_verification else "ssd_data_written"
        child = self.manager(path, pauses=[] if during_verification else [point])
        child.send("write", id=1, ttl_millis=8000)
        if during_verification:
            assert child.reply()["present"]
            child.close(kill=True)
            child = self.manager(path, pauses=[point])
        child.reached(point)
        records = [
            value for value in frames(Path(str(path) + ".index")) if value.get("expiry")
        ]
        assert records, "publication lifetime must already be durable"
        expires = records[-1]["expiry"]["expires_unix_millis"]
        child.wait(lambda: time.time_ns() // 1_000_000 >= expires)
        child.release(point)
        if during_verification:
            child.recovered()
            child.read(present=False)
        else:
            assert not child.reply()["present"], (
                "late write renewed its expired publication"
            )
        child.close(kill=True)
        child = self.manager(path)
        child.read(present=False)
        child.close()

    def old_file_user(self, path):
        child = self.manager(path)
        assert child.call("write", id=1)["present"]
        # Independent file description stands in for a kernel-retained file
        # user after the Manager's own flock is gone. No receipt is invented.
        with path.open("rb") as outstanding:
            child.close(kill=True)
            before = Path(str(path) + ".index").read_bytes()
            rejected = self.manager(path, reject_with="drained prior file users")
            rejected.close()
            assert Path(str(path) + ".index").read_bytes() == before
            assert outstanding.read(137) == bytes([1]) * 137
        child = self.manager(path)
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

    def tail_prefix(self, path, kind):
        child = self.manager(path)
        assert child.call("write", id=1)["present"]
        if kind == "kill":
            child.pause("ssd_mid_record")
            child.send("write", id=2, encoded=True)
            child.reached("ssd_mid_record")
            child.close(kill=True)
        else:
            assert child.call("write", id=2, encoded=True)["present"]
            child.close()
            journal = Path(str(path) + ".index")
            data = bytearray(journal.read_bytes())
            if kind == "truncate":
                data = data[:-1]
            else:
                # Header, first fence, first commit, then a valid fence/commit.
                offset = 0
                for _ in range(2):
                    length = int.from_bytes(data[offset : offset + 4], "little")
                    offset += 8 + length
                data[offset + 4] ^= 1
            journal.write_bytes(data)
        child = self.manager(path)
        result = child.read(1, present=kind != "mid")
        child.read(2, present=False)
        report = result["state"]["recovery"]
        if kind == "mid":
            assert report["drops"]["metadata_mid_journal"] == 1, report
        else:
            assert report["tail_dropped_bytes"] > 0, report
        assert child.call("write", id=3)["present"]
        child.close()
        child = self.manager(path)
        child.read(1, present=kind != "mid")
        child.read(3)
        assert not child.call("report")["state"]["recovery"]["drops"]
        child.close()

    def read_integrity(self, path, persistent):
        from urllib.request import urlopen

        child = self.manager(path, persistent=persistent)
        for id, (encoded, padding) in enumerate(
            [(False, False), (True, False), (False, True), (True, True)], start=1
        ):
            assert child.call("write", id=id, encoded=encoded)["present"]
            child.read(id, encoded=encoded)
            if persistent:
                child.close()
                child = self.manager(path)
                child.read(
                    id, encoded=encoded
                )  # Recovery verified this extent already.
            with path.open("r+b") as file:
                file.seek((id - 1) * 1024 + (900 if padding else 17))
                file.write(b"\xff")
                os.fsync(file.fileno())
            missed = child.read(id, present=False)
            assert "slots" not in missed, "corrupt bytes must not reach a consumer"
            assert not child.call("read", id=id)["present"], (
                "bad generation must be evicted"
            )
            with urlopen(f"http://{child.http}/metrics", timeout=10) as response:
                metrics = response.read().decode()
            failures = [
                line
                for line in metrics.splitlines()
                if line.startswith("orbitkv_integrity_failures_total{")
                and 'tier="ssd"' in line
                and 'reason="checksum"' in line
            ]
            assert failures and sum(float(line.split()[-1]) for line in failures) == (
                1 if persistent else id
            ), metrics
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

    def watermark_blocked(self, child):
        from urllib.request import urlopen

        def blocked():
            with urlopen(f"http://{child.http}/metrics", timeout=10) as response:
                lines = response.read().decode().splitlines()
            return any(
                line.split("{", 1)[0].split(" ", 1)[0]
                == "orbitkv_ssd_watermark_blocked_total"
                and float(line.split()[-1]) > 0
                for line in lines
            )

        child.wait(blocked)

    def watermark_reader(self, path, receipt):
        child = self.manager(path, watermarks=(25, 75))
        assert child.call("write", id=1)["present"]
        if receipt:
            child.pause("ssd_read_receipt")
            child.call("read_start", id=1)
            child.reached("ssd_read_receipt")
        else:
            assert child.call("hold", id=1)["present"]
        for id in (2, 3):
            assert child.call("write", id=id)["present"]
        self.watermark_blocked(child)
        state = child.call("report")["state"]["ring"]
        assert state["entries"] == 3 and state["readers"] > 0, state
        if receipt:
            child.release("ssd_read_receipt")
            result = child.call("read_finish")
            assert [slot["bytes"] for slot in result["slots"]] == [[1] * 137, [2] * 631]
        else:
            child.call("release")
        child.wait(lambda: child.call("report")["state"]["ring"]["entries"] == 1)
        child.read(1, present=False)
        child.read(2, present=False)
        child.read(3)
        child.close(kill=True)
        restarted = self.manager(path)
        restarted.read(1, present=False)
        restarted.read(2, present=False)
        restarted.read(3)
        restarted.close()

    def watermark_writing(self, path):
        child = self.manager(
            path, capacity=1024, watermarks=(0, 100), pauses=["ssd_data_written"]
        )
        child.send("write", id=1)
        child.reached("ssd_data_written")
        self.watermark_blocked(child)
        child.release("ssd_data_written")
        child.reply()  # Visibility may already be removed by the next maintenance tick.
        child.wait(lambda: child.call("report")["state"]["ring"]["entries"] == 0)
        child.close(kill=True)
        restarted = self.manager(path, capacity=1024)
        restarted.read(1, present=False)
        restarted.close()

    def watermark_crash(self, path):
        child = self.manager(path)
        for id in (1, 2, 3):
            assert child.call("write", id=id)["present"]
        child.close()
        child = self.manager(
            path, watermarks=(25, 75), pauses=["ssd_watermark_invalidated"]
        )
        child.reached("ssd_watermark_invalidated")
        child.close(kill=True)
        restarted = self.manager(path)
        restarted.read(1, present=False)
        restarted.read(2)
        restarted.read(3)
        restarted.close()

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
    gate.case("concurrent-shared-namespaces", gate.shared_namespaces)
    gate.case("shared-partition-reopen", gate.shared_partition_reopen)
    gate.case("shared-root-capacity", gate.shared_root_capacity)
    gate.case("reclaim-offline-shared-partition", gate.reclaim_offline_partition)
    gate.case("reclaim-live-shared-partition", gate.relinquish_live_partition)
    gate.case("ttl-late-write", lambda path: gate.ttl(path, False))
    gate.case("ttl-during-recovery-verification", lambda path: gate.ttl(path, True))
    gate.case("prior-file-user-blocks-restart", gate.old_file_user)
    for kind in ["data", "torn", "checksum", "version", "namespace"]:
        gate.case(f"corrupt-{kind}", lambda path, k=kind: gate.corrupt(path, k))
    for kind in ["kill", "truncate", "mid"]:
        gate.case(f"prefix-{kind}", lambda path, k=kind: gate.tail_prefix(path, k))
    for persistent in [False, True]:
        gate.case(
            f"read-integrity-{persistent}",
            lambda path, p=persistent: gate.read_integrity(path, p),
        )
    for point in ["ssd_reserved", "ssd_data_written"]:
        gate.case(f"wrap-{point}", lambda path, p=point: gate.wrap(path, p))
    gate.case("reader-protects-reuse", lambda path: gate.owners(path, False))
    gate.case("receipt-protects-reuse", lambda path: gate.owners(path, True))
    gate.case("watermark-held-reader", lambda path: gate.watermark_reader(path, False))
    gate.case("watermark-read-receipt", lambda path: gate.watermark_reader(path, True))
    gate.case("watermark-writing", gate.watermark_writing)
    gate.case("watermark-invalidation-crash", gate.watermark_crash)
    gate.case("capacity-shrink", gate.shrink)
    for point in ["ssd_checkpoint_synced", "ssd_checkpoint_replaced"]:
        gate.case(point, lambda path, p=point: gate.checkpoint(path, p))
    gate.case("disk-full", gate.disk_full)
    print(f"{len(gate.results)} required cases passed ({args.filesystem})")


if __name__ == "__main__":
    main()
