"""Real CPU processes protect the RDMA harness's admission and stop boundaries."""

import contextlib
import errno
import fcntl
import hashlib
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from benches.distributed.rdma_owner import HostOwner
from benches.distributed.rdma_profile import Campaign, HostConnection
from benches.distributed.tent_gpu import CAPACITY, slices, validate_stop


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
from benches.distributed.tent_gpu import retain
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


def attach_runtime_inputs(contract, host, tmp_path):
    helper = Path(__file__).resolve().parents[3] / "python/tests/support/registration_campaign.py"
    resources = contract.setdefault("resource_locks", [])
    declarations = []
    for name in resources:
        path = Path(name)
        path.touch(exist_ok=True)
        metadata = path.stat()
        declarations.append({"path": name, "device": metadata.st_dev, "inode": metadata.st_ino})
    guard = tmp_path / "runtime-guard.json"
    guard.write_text(
        json.dumps(
            {
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "input_sha256": {},
                "runtime_manifests": [],
                "protected_resources": [],
                "locks": declarations,
                "channel_domains": [],
            }
        )
    )
    contract["runtime_inputs"] = {host: {"helper": str(helper), "guard": str(guard)}}
    contract["runtime_hashes"] = {
        host: {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (helper, guard)}
    }
    return guard


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
        "resource_locks": [str(tmp_path / "gpu.lock")],
        "file_stamps": {"local": {}},
        "launches": {
            "local": {
                "cell": {
                    "gpu-reader": {
                        "argv": command,
                        "environment": {"PYTHONPATH": str(Path(__file__).resolve().parents[3])},
                        "kind": "gpu",
                    }
                }
            }
        },
    }
    attach_runtime_inputs(contract, "local", tmp_path)
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

    monkeypatch.setattr("benches.distributed.rdma_owner.write", no_space)
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

    monkeypatch.setattr("benches.distributed.rdma_owner.snapshot", failed_snapshot)
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
    root = Path(__file__).resolve().parents[3]
    contract = {
        "hosts": {
            "local": {
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "gpu_uuids": {"7": "cpu-test"},
            }
        },
        "resource_locks": [str(tmp_path / "gpu.lock")],
        "runtime_hashes": {"local": {}},
        "file_stamps": {"local": {}},
        "launches": {"local": {}},
    }
    attach_runtime_inputs(contract, "local", tmp_path)
    config = tmp_path / "CONTRACT.json"
    config.write_text(json.dumps(contract))
    index = tmp_path / "INPUTS.json"
    index.write_text(json.dumps({"CONTRACT.json": hashlib.sha256(config.read_bytes()).hexdigest()}))
    (tmp_path / "LAUNCH-ACCEPTED.json").write_text(
        json.dumps(
            {"accepted": True, "inputs_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
        )
    )
    program = """
from benches.distributed import rdma_owner
rdma_owner.snapshot=lambda:{'boot_id':rdma_owner.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'gpu':'7, cpu-test, 0\\n','compute':''}
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

        monkeypatch.setattr("benches.distributed.rdma_profile.save", no_space)
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


@pytest.mark.parametrize("foreign_listener", [False, True])
def test_service_status_requires_the_actual_owned_listening_socket(tmp_path, foreign_listener):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    if foreign_listener:
        program = "import time; time.sleep(30)"
    else:
        listener.close()
        program = (
            "import socket,time; s=socket.socket(); "
            "s.bind(('127.0.0.1',int(__import__('sys').argv[1]))); s.listen(); time.sleep(30)"
        )
    owner = service_owner(tmp_path, [sys.executable, "-u", "-c", program, str(port)])
    try:
        deadline = time.monotonic() + 5
        while True:
            result = owner.process_status({"name": "manager"})
            assert result["exit_code"] is None and not result["actual_wait_reap"]
            assert result["host"] == "local"
            assert result["boot_id"] == owner.contract["hosts"]["local"]["boot_id"]
            if foreign_listener or port in result["listen_ports"]:
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert (port in result["listen_ports"]) is not foreign_listener
    finally:
        listener.close()
        receipt = owner.stop({"name": "manager", "idle_proven": True})
        assert receipt["exit_code"] == -signal.SIGTERM and receipt["actual_wait_reap"]


def service_owner(tmp_path, argv, environment=None):
    environment = {} if environment is None else environment
    contract = {
        "hosts": {
            "local": {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
        },
        "file_stamps": {"local": {}},
        "launches": {
            "local": {
                "startup": {
                    "manager": {"argv": argv, "environment": environment, "kind": "service"}
                }
            }
        },
    }
    attach_runtime_inputs(contract, "local", tmp_path)
    owner = HostOwner(contract, "local", tmp_path)
    owner.launch({"cell": "startup", "name": "manager", "argv": argv, "environment": environment})
    return owner


@pytest.mark.parametrize("boundary", ["launch", "finish"])
@pytest.mark.parametrize("mutation", ["extra_module", "replaced_channel_directory"])
def test_runtime_change_blocks_the_actual_owner_boundary(monkeypatch, tmp_path, boundary, mutation):
    owner, lease = cpu_owner(tmp_path)
    try:
        owner.call({"name": "gpu-reader", "request": {"operation": "stop", "id": "stop"}})
        owner.stop({"name": "gpu-reader"})
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        payload = runtime / "frozen.py"
        payload.write_text("value = 1\n")
        manifest = tmp_path / "runtime-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "root": str(runtime),
                    "files": [
                        {
                            "path": payload.name,
                            "realpath": str(payload),
                            "bytes": payload.stat().st_size,
                            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                        }
                    ],
                }
            )
        )
        channel = tmp_path / "channel"
        channel.mkdir()
        (channel / "nodes").mkdir()
        (channel / "services").mkdir()
        directories = []
        for path in (channel, channel / "nodes", channel / "services"):
            metadata = path.stat()
            directories.append(
                {
                    "path": str(path),
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "mode": metadata.st_mode,
                    "uid": metadata.st_uid,
                }
            )
        guard_path = Path(owner.contract["runtime_inputs"]["local"]["guard"])
        guard = json.loads(guard_path.read_text())
        guard["runtime_manifests"] = [
            {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        ]
        guard["channel_domains"] = [
            {"cell": "cell", "root": str(channel), "directories": directories}
        ]
        owner.contract["cells"] = [{"id": "cell", "kind": "manager"}]
        guard_path.write_text(json.dumps(guard))
        owner.contract["runtime_hashes"]["local"][str(guard_path)] = hashlib.sha256(
            guard_path.read_bytes()
        ).hexdigest()
        owner.validate_runtime()
        if mutation == "extra_module":
            (runtime / "unexpected.py").write_text("value = 2\n")
            failure = "Runtime file census"
        else:
            (channel / "nodes").rename(channel / "retired-nodes")
            (channel / "nodes").mkdir()
            failure = "Private runtime directory identity"
        if boundary == "launch":
            planned = owner.contract["launches"]["local"]["cell"]["gpu-reader"]
            with pytest.raises(RuntimeError, match=failure):
                owner.launch(
                    {
                        "cell": "cell",
                        "name": "gpu-reader",
                        "argv": planned["argv"],
                        "environment": planned["environment"],
                    }
                )
        else:
            monkeypatch.setattr(
                "benches.distributed.rdma_owner.snapshot",
                lambda: {"gpu": "3, cpu, 0\n", "compute": ""},
            )
            with pytest.raises(RuntimeError, match=failure):
                owner.finish()
            assert owner.uncertain
        assert owner.children == {}
        with Path(lease.name).open("a") as contender, pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        for name in list(owner.children):
            owner.call({"name": name, "request": {"operation": "stop", "id": "cleanup"}})
            owner.stop({"name": name})
        lease.close()


def test_child_receives_only_the_frozen_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LD_AUDIT", "/missing-cpu-control-audit.so")
    monkeypatch.setenv("PYTHONINSPECT", "1")
    program = (
        "import json,os,time; print(json.dumps({'ld_audit':'LD_AUDIT' in os.environ,"
        "'inspect':'PYTHONINSPECT' in os.environ,'input':os.environ.get('S4_CHILD_INPUT')}),flush=True);"
        "time.sleep(30)"
    )
    owner = service_owner(
        tmp_path, [sys.executable, "-u", "-c", program], {"S4_CHILD_INPUT": "frozen"}
    )
    try:
        path = tmp_path / "startup-manager.stdout.log"
        deadline = time.monotonic() + 5
        while not path.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert json.loads(path.read_text()) == {
            "ld_audit": False,
            "inspect": False,
            "input": "frozen",
        }
    finally:
        owner.stop({"name": "manager", "idle_proven": True})


@pytest.mark.parametrize("wrong_origin", [False, True])
def test_actual_owned_runtime_checkpoint_consumes_manager_origin(tmp_path, wrong_origin):
    origins = {
        "torch_file": "/cpu-control/torch/__init__.py",
        "prefix": "/cpu-control",
        "base_prefix": "/cpu-control",
        "allowed_sys_path": ["/cpu-control"],
        "gpu_module_file": "/cpu-control/orbitkv/client/gpu.py",
    }
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    program = """
import json,os,socket,sys,time
role=sys.argv[1];expected=json.loads(sys.argv[2])
if role=='manager':
    runtime={key:expected[key] for key in ('torch_file','prefix','base_prefix')}
    runtime.update(pid=os.getpid(),sys_path=expected['allowed_sys_path'],no_user_site=1)
    print('CUDA registration Python runtime: '+json.dumps(runtime),flush=True)
    print('CUDA registration Python helper: '+json.dumps({'pid':os.getpid(),'gpu_module_file':expected['gpu_module_file']}),flush=True)
listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen()
time.sleep(30)
"""
    contract = {
        "hosts": {
            "local": {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
        },
        "launches": {"local": {"cell": {}}},
        "file_stamps": {"local": {}},
        "python_origins": {"local": origins},
        "observer_library": {"local": str(tmp_path / "unused-observer.so")},
    }
    guard_path = attach_runtime_inputs(contract, "local", tmp_path)
    observed = dict(origins)
    if wrong_origin:
        observed["gpu_module_file"] = "/cpu-control/foreign.py"
    for name in ("manager", "fixture"):
        contract["launches"]["local"]["cell"][name] = {
            "argv": [sys.executable, "-u", "-c", program, name, json.dumps(observed)],
            "environment": environment,
            "kind": "service",
            "runtime": {
                "executable": str(Path(sys.executable).resolve()),
                "required_mapped_files": [str(Path(sys.executable).resolve())],
                "observer_enabled": False,
            },
        }
    owner = HostOwner(contract, "local", tmp_path)
    try:
        for name in ("manager", "fixture"):
            planned = contract["launches"]["local"]["cell"][name]
            owner.launch(
                {
                    "cell": "cell",
                    "name": name,
                    "argv": planned["argv"],
                    "environment": planned["environment"],
                }
            )
        deadline = time.monotonic() + 5
        while any(
            not owner.process_status({"name": name})["listen_ports"] for name in owner.children
        ):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        checks, guard = owner.runtime_checks()
        for item in owner.children.values():
            captured = checks["capture_process_runtime"](
                item["process"].pid, item["start_ticks"], time.monotonic() + 10
            )
            guard["input_sha256"].update(
                {row["realpath"]: row["sha256"] for row in captured["files"]}
            )
        guard_path.write_text(json.dumps(guard))
        contract["runtime_hashes"]["local"][str(guard_path)] = hashlib.sha256(
            guard_path.read_bytes()
        ).hexdigest()
        path = tmp_path / "cell-runtime-admission.json"
        if wrong_origin:
            with pytest.raises(RuntimeError, match="Python importer"):
                owner.admit_runtime({"cell": "cell"})
            assert not path.exists()
        else:
            receipt = owner.admit_runtime({"cell": "cell"})
            assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert set(receipt["owners"]) == {"manager", "fixture"}
            assert not receipt["physical_termination_claimed"]
    finally:
        for name in list(owner.children):
            owner.stop({"name": name, "idle_proven": True})


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_manager_bootstrap_failure_preserves_primary_error_and_reaps_real_child(
    monkeypatch, tmp_path, cleanup_failure
):
    owner = service_owner(
        tmp_path, [sys.executable, "-u", "-c", "raise SystemExit(17)", "--addr", "127.0.0.1:1"]
    )
    campaign = Campaign(
        {"launches": {"local": {"startup": owner.contract["launches"]["local"]["startup"]}}},
        tmp_path,
    )

    class LocalConnection:
        uncertain = False

        def rpc(self, operation, **fields):
            if operation == "snapshot":
                return {"cpu_only_control": True}
            if operation == "process-status":
                return owner.process_status(fields)
            if operation == "stop":
                return owner.stop(fields)
            raise AssertionError(operation)

    campaign.hosts["local"] = LocalConnection()
    monkeypatch.setattr(campaign, "launch", lambda *args: None)

    def membership_check(cell):
        if cleanup_failure:
            raise ConnectionError("controlled unavailable etcd during cleanup")

    monkeypatch.setattr(campaign, "members_absent", membership_check)

    def forbidden_drain(*args):
        pytest.fail("A fresh cell without a fixture must not contact its dead Manager for drain")

    monkeypatch.setattr("benches.distributed.rdma_profile.drain", forbidden_drain)
    campaign.active = {"local": {"manager": "startup"}}
    output = tmp_path / "failed-cell.json"
    try:
        with pytest.raises(ChildProcessError, match="Owned service exited before readiness"):
            campaign.manager_cell(
                {"id": "startup", "manager_urls": {"local": "http://127.0.0.1:1"}}, output
            )
        result = json.loads(output.read_text())
        assert result["status"] == "INVALID_OR_FAILED"
        assert result["error_type"] == "ChildProcessError"
        assert "17" in result["error"] and "Owned service exited" in result["traceback"]
        assert result["exits"][0]["exit_code"] == 17
        assert result["exits"][0]["actual_wait_reap"]
        assert ("cleanup_error" in result) is cleanup_failure
        assert owner.children == {} and campaign.active == {"local": {}}
    finally:
        if "manager" in owner.children:
            owner.stop({"name": "manager", "idle_proven": True})
