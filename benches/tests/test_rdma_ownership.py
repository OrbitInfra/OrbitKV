"""Real CPU processes protect the RDMA harness's admission and stop boundaries."""

import contextlib
import errno
import fcntl
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benches.rdma_owner import HostOwner
from benches.rdma_profile import HostConnection
from benches.tent_gpu import CAPACITY, slices, validate_stop


@pytest.mark.parametrize(
    "payload,chunk,stride,count", [(32 << 20, 32 << 20, 32 << 20, 1), (32 << 20, 4096, 8192, 8192)]
)
def test_gpu_slices_cover_only_the_declared_payload(payload, chunk, stride, count):
    result = slices(1 << 30, 2 << 30, payload, chunk, stride)
    assert len(result) == count and sum(row[2] for row in result) == payload
    assert result[-1][0] + chunk <= (1 << 30) + CAPACITY
    assert result[-1][1] + chunk <= (2 << 30) + CAPACITY


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 2, 4096, 4096, 4096),
        (1, 2, True, 4096, 4096),
        (1, 2, 32 << 20, 4096, 8193),
        (1, 2, 32 << 20, 4096, 16384),
        (1, 2, 4097, 4096, 4096),
        (2**64 - 100, 2, 4096, 4096, 4096),
    ],
)
def test_gpu_slices_reject_nonregistered_or_unaligned_geometry(arguments):
    with pytest.raises(ValueError):
        slices(*arguments)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("cell", "previous"),
        ("host", "other-host"),
        ("name", "gpu-peer"),
        ("native_stopped", False),
        ("actual_wait_reap", False),
        ("forced_cleanup", True),
        ("exit_code", -9),
        ("exit_code", True),
    ],
)
def test_peer_gpu_stop_requires_same_cell_actual_reader_reap(field, bad):
    receipt = {
        "cell": "current",
        "host": "h1",
        "name": "gpu-reader",
        "native_stopped": True,
        "actual_wait_reap": True,
        "forced_cleanup": False,
        "exit_code": 0,
    }
    validate_stop({"peer_receipt": receipt}, "peer", "current", "h1")
    receipt[field] = bad
    with pytest.raises(ValueError):
        validate_stop({"peer_receipt": receipt}, "peer", "current", "h1")


CPU_ENDPOINT = """
import json, os, signal, sys
from benches.tent_gpu import retain
for sig in [signal.SIGHUP, signal.SIGTERM, signal.SIGINT]:signal.signal(sig, signal.SIG_IGN)
fd=int(sys.argv[-1]);os.fstat(fd)
print(json.dumps({'tent_gpu_endpoint':True,'event':'ready','leased_fd':fd}),flush=True)
for line in sys.stdin:
    message=json.loads(line)
    if message['operation']=='stop':
        print(json.dumps({'tent_gpu_endpoint':True,'event':'stopped','native_stopped':True,
                          'registered_regions':0,'gpu_allocations':0}),flush=True)
        break
else:retain('CPU ownership control EOF; no CUDA or native memory in this test')
"""


def cpu_owner(tmp_path):
    out = tmp_path / "owner"
    out.mkdir()
    lease = (tmp_path / "gpu.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = [sys.executable, "-u", "-c", CPU_ENDPOINT]
    contract = {
        "hosts": {
            "local": {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
        },
        "file_stamps": {"local": {}},
        "launches": {
            "local": {
                "cell": {
                    "gpu-reader": {
                        "argv": command,
                        "environment": {"PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                        "kind": "gpu",
                    }
                }
            }
        },
    }
    owner = HostOwner(contract, "local", out)
    owner.leases = [lease]
    owner.launch(
        {
            "name": "gpu-reader",
            "cell": "cell",
            "argv": command,
            "environment": contract["launches"]["local"]["cell"]["gpu-reader"]["environment"],
        }
    )
    ready = owner.call({"name": "gpu-reader"})
    assert ready["event"] == "ready"
    return owner, lease


@pytest.mark.parametrize("disconnect", [False, True])
def test_real_child_keeps_inherited_lease_until_proven_exit(tmp_path, disconnect):
    owner, lease = cpu_owner(tmp_path)
    process = owner.children["gpu-reader"]["process"]
    contender = (tmp_path / "gpu.lock").open("a")
    try:
        lease.close()
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="native drain"):
            owner.stop({"name": "gpu-reader"})
        if disconnect:
            process.stdin.close()
            event = owner.children["gpu-reader"]["queue"].get(timeout=5)
            assert event["event"] == "ownership_unknown" and event["ownership_held"]
            process.send_signal(signal.SIGHUP)
            process.send_signal(signal.SIGTERM)
            time.sleep(0.05)
            assert process.poll() is None
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            owner.call({"name": "gpu-reader", "request": {"operation": "stop", "id": "stop"}})
            result = owner.stop({"name": "gpu-reader"})
            assert result["exit_code"] == 0 and result["actual_wait_reap"]
            assert result["native_stopped"] and not result["forced_cleanup"]
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        # The deliberately retained process is CPU-only and never exported native memory.
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        item = owner.children.pop("gpu-reader", None)
        if item:
            item["thread"].join(timeout=5)
            for stream in (process.stdin, process.stdout, item["stdout"], item["stderr"]):
                if not stream.closed:
                    stream.close()
        contender.close()
        lease.close()


def test_quarantine_enospc_retains_lease_and_blocks_admission(monkeypatch, tmp_path):
    owner = HostOwner({}, "cpu-test", tmp_path)
    lease = (tmp_path / "gpu.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner.leases = [lease]

    def no_space(*args):
        raise OSError(errno.ENOSPC, "controlled quarantine ENOSPC")

    monkeypatch.setattr("benches.rdma_owner.write", no_space)
    try:
        owner.quarantine("unknown native state")
        assert owner.uncertain
        with pytest.raises(RuntimeError, match="Quarantined"):
            owner.launch({})
        with (tmp_path / "gpu.lock").open("a") as contender, pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        lease.close()


def test_physical_postflight_failure_keeps_real_resource_lease(monkeypatch, tmp_path):
    owner, lease = cpu_owner(tmp_path)
    owner.call({"name": "gpu-reader", "request": {"operation": "stop", "id": "stop"}})
    result = owner.stop({"name": "gpu-reader"})
    assert result["actual_wait_reap"] and result["exit_code"] == 0

    def failed_snapshot():
        raise OSError("controlled unavailable physical inspection")

    monkeypatch.setattr("benches.rdma_owner.snapshot", failed_snapshot)
    try:
        with pytest.raises(OSError, match="physical inspection"):
            owner.finish()
        assert owner.uncertain and owner.work_admitted
        assert (owner.output / "OWNERSHIP-UNKNOWN.json").exists()
        with (tmp_path / "gpu.lock").open("a") as contender, pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        lease.close()


def test_real_host_owner_retains_lease_on_controller_eof(tmp_path):
    root = Path(__file__).resolve().parents[2]
    contract = {
        "hosts": {"local": {"boot_id": "cpu-test", "gpu_uuids": {"7": "cpu-test"}}},
        "resource_locks": [str(tmp_path / "gpu.lock")],
        "runtime_hashes": {"local": {}},
        "file_stamps": {"local": {}},
        "launches": {"local": {}},
    }
    config = tmp_path / "CONTRACT.json"
    config.write_text(json.dumps(contract))
    import hashlib

    index = tmp_path / "INPUTS.json"
    index.write_text(json.dumps({"CONTRACT.json": hashlib.sha256(config.read_bytes()).hexdigest()}))
    (tmp_path / "LAUNCH-ACCEPTED.json").write_text(
        json.dumps(
            {"accepted": True, "inputs_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
        )
    )
    program = """
from benches import rdma_owner
rdma_owner.snapshot=lambda:{'boot_id':'cpu-test','gpu':'7, cpu-test, 0\\n','compute':''}
rdma_owner.main()
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            program,
            "--contract",
            str(config),
            "--host",
            "local",
            "--output",
            str(tmp_path / "out"),
        ],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert json.loads(process.stdout.readline())["ready"]
        process.stdin.close()
        deadline = time.monotonic() + 5
        while (
            not (tmp_path / "out/OWNERSHIP-UNKNOWN.json").exists() and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert (tmp_path / "out/OWNERSHIP-UNKNOWN.json").exists()
        process.send_signal(signal.SIGHUP)
        process.send_signal(signal.SIGTERM)
        time.sleep(0.05)
        assert process.poll() is None
        with (tmp_path / "gpu.lock").open("a") as contender, pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        # This isolated owner has no child, GPU, registered address or outstanding transfer.
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        with contextlib.suppress(OSError):
            process.stdout.close()
        with contextlib.suppress(OSError):
            process.stderr.close()


@pytest.mark.parametrize("failure", ["bad_ready", "receipt_enospc"])
def test_controller_registers_real_owner_before_ready_can_fail(monkeypatch, tmp_path, failure):
    program = """
import fcntl,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);lock=(root/'controller.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
print(json.dumps({'rdma_owner':True,'host':'local','ready':sys.argv[2]=='receipt_enospc'}),flush=True)
while not (root/'cpu-only-release').exists():time.sleep(.01)
"""
    connections = {}
    if failure == "receipt_enospc":

        def no_space(*args):
            raise OSError(errno.ENOSPC, "controlled ready receipt ENOSPC")

        monkeypatch.setattr("benches.rdma_profile.save", no_space)
    connections["local"] = HostConnection(
        [sys.executable, "-u", "-c", program, str(tmp_path), failure], "local", tmp_path
    )
    owner = connections["local"]
    try:
        with pytest.raises((RuntimeError, OSError)):
            owner.start()
        assert connections["local"] is owner and owner.uncertain
        assert owner.process.returncode is None and owner.process.poll() is None
        with (tmp_path / "controller.lock").open("a") as contender, pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        # This cooperating fixture owns only a CPU flock; it has no native export.
        (tmp_path / "cpu-only-release").touch()
        assert owner.process.wait(timeout=5) == 0
        owner.thread.join(timeout=5)
        for stream in (owner.process.stdin, owner.process.stdout, owner.stdout, owner.stderr):
            stream.close()
