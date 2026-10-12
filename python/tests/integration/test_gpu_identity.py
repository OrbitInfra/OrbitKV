"""Real registration/restore under independently filtered CUDA process views.

Trigger: native GPU identity, registration routing or NUMA placement changes.
Requires two dedicated CUDA GPUs and an installed wheel/Manager. Set
ORBITKV_GPU_IDENTITY_DEVICES to their physical indices, target first, and
ORBITKV_GPU_IDENTITY_OUTPUT to an external evidence directory.
"""

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.support.cache_manager import CacheManagerProcess, find_available_port
from tests.support.gpu_identity import (
    drain_identity_processes,
    identity_gpu_locks,
    owned_process,
    quarantine_identity_gpus,
    require_clean_identity_log,
)
from tests.support.metrics import fetch_orbitkv_metrics

pytestmark = pytest.mark.integration

PROFILES = [
    ("numeric", "target", "target", 0, 0, ""),
    ("uuid", "target_uuid", "target", 0, 0, ""),
    ("manager_reordered", "other,target", "target,other", 0, 1, ""),
    ("manager_uuid_reordered", "other_uuid,target_uuid", "target_uuid", 0, 1, ""),
    ("client_reordered", "target,other", "other,target", 1, 0, ""),
    ("partial", "target_uuid", "other,target", 1, 0, ""),
]
CASES = [(*profile, medium) for profile in PROFILES for medium in ("dram", "ssd")] + [
    ("unseen", "other_uuid", "target_uuid", 0, None, "not found", "dram"),
    ("mixed_wrapper", "target,other", "target,other", 0, None, "UUID differs", "dram"),
    ("wrong_ordinal", "target_uuid", "target_uuid", 0, None, "client-local", "dram"),
]


@pytest.fixture(scope="module")
def identity_devices():
    selected = os.environ.get("ORBITKV_GPU_IDENTITY_DEVICES", "").split(",")
    if len(selected) != 2 or not os.environ.get("ORBITKV_GPU_IDENTITY_OUTPUT"):
        pytest.skip("Select two dedicated GPUs and an external identity evidence directory")
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    )
    uuids = dict(line.strip().split(", ") for line in query.strip().splitlines())
    assert selected[0] != selected[1]
    for device in selected:
        assert device in uuids, device
    output = Path(os.environ["ORBITKV_GPU_IDENTITY_OUTPUT"])
    output.mkdir(parents=True, exist_ok=False)
    lock_root = Path(os.environ.get("ORBITKV_TEST_GPU_LOCK_DIR", str(output.parent / "gpu-locks")))
    with identity_gpu_locks(lock_root, selected) as lock_fds:

        def gpu_snapshot():
            observed = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            memory = {
                uuid: int(used)
                for uuid, used in (line.split(", ") for line in observed.splitlines())
            }
            return {uuids[device]: memory[uuids[device]] for device in selected}

        before = gpu_snapshot()
        (output / "gpu-preflight.json").write_text(json.dumps(before, indent=2) + "\n")
        assert not any(before.values()), before
        values = {
            "target": selected[0],
            "other": selected[1],
            "target_uuid": uuids[selected[0]],
            "other_uuid": uuids[selected[1]],
            "output": output,
            "lock_root": lock_root,
            "devices": selected,
            "lock_fds": lock_fds,
            "failed": False,
        }
        try:
            yield values
        finally:
            deadline = time.monotonic() + 10
            while True:
                after = gpu_snapshot()
                if not any(after.values()) or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            (output / "gpu-postflight.json").write_text(json.dumps(after, indent=2) + "\n")
            if any(after.values()):
                quarantine_identity_gpus(
                    lock_root,
                    selected,
                    output,
                    {"remaining_processes": [{"gpu_memory_mib": after}], "errors": []},
                )
            assert not any(after.values()), after


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-{c[-1]}" for c in CASES])
def test_gpu_identity_restore_and_numa(identity_devices, monkeypatch, case):
    name, manager_mask, client_mask, client_device, manager_device, rejection, medium = case
    values = identity_devices
    if values["failed"]:
        pytest.skip("Not launched: an earlier identity cell failed")
    directory = values["output"] / f"{name}-{medium}"
    directory.mkdir()
    manager_visible = ",".join(values[part] for part in manager_mask.split(","))
    client_visible = ",".join(values[part] for part in client_mask.split(","))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", manager_visible)
    port, http_port = find_available_port(), find_available_port()
    server = CacheManagerProcess(
        port,
        pool_size="128mb",
        devices=",".join(str(ordinal) for ordinal in range(len(manager_visible.split(",")))),
        http_port=http_port,
        bootstrap_socket=f"/tmp/okv-id-{port}.sock",
        ssd_cache_path=directory / "ssd" if medium == "ssd" else None,
        log_path=directory / "manager.log",
        runtime_python_paths=tuple(
            path for path in os.environ.get("PYTHONPATH", "").split(":") if path
        )
        + tuple(path for path in sys.path if Path(path).name == "site-packages"),
        inherited_fds=values["lock_fds"],
        extra_args=("--enable-prometheus",) if medium == "dram" else ("--ssd-write-policy", "all"),
    )
    result = {
        "manager_visible": manager_visible,
        "client_visible": client_visible,
        "client_device": client_device,
        "manager_device": manager_device,
        "rejection": rejection,
        "medium": medium,
        "forced_cleanup": False,
        "ports": [port, http_port],
        "bootstrap_socket": server.bootstrap_socket,
    }
    client = None
    completed = False
    cleanup = None
    worker_output = directory / "worker-result.json"
    try:
        assert server.start(), server.read_logs()
        result.update(manager_pid=server.process.pid, manager_command=server.command)
        result["manager_identity"] = owned_process(server.process)
        (directory / "manager-maps.txt").write_text(
            Path(f"/proc/{server.process.pid}/maps").read_text()
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=client_visible)
        with (directory / "client.log").open("wb") as log:
            client = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "tests.support.gpu_identity_worker",
                    "--socket",
                    server.bootstrap_socket,
                    "--http-port",
                    str(http_port),
                    "--device",
                    str(client_device),
                    "--uuid",
                    values["target_uuid"],
                    "--output",
                    str(worker_output),
                    "--medium",
                    medium,
                    "--rejection",
                    rejection,
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=values["lock_fds"],
                start_new_session=True,
            )
        result["client_pid"] = client.pid
        result["client_identity"] = owned_process(client)
        deadline = time.monotonic() + 90
        while not worker_output.with_suffix(".ready.json").exists():
            assert client.poll() is None, (directory / "client.log").read_text()
            assert time.monotonic() < deadline, "Client readiness timeout"
            time.sleep(0.05)
        if not rejection:
            nearby = subprocess.check_output(
                ["nvidia-smi", "topo", "--get-numa-id-of-nearby-cpu", "-i", values["target_uuid"]],
                text=True,
            )
            numa = int(nearby.split(":", 1)[1].strip().split(",")[0])
            result["numa_node"] = numa
            assert (
                f"CUDA device {manager_device} ({values['target_uuid']}) -> NUMA{numa}"
                in server.read_logs()
            )
            expected_cpus = Path(f"/sys/devices/system/node/node{numa}/cpulist").read_text().strip()
            workers = {}
            for thread in Path(f"/proc/{server.process.pid}/task").iterdir():
                comm = (thread / "comm").read_text().strip()
                if comm in {f"gpu{manager_device}-load", f"gpu{manager_device}-save"}:
                    status = (thread / "status").read_text()
                    workers[comm] = re.search(r"Cpus_allowed_list:\s*(.+)", status).group(1).strip()
            assert len(workers) == 2, workers
            assert set(workers.values()) == {expected_cpus}, (workers, expected_cpus)
            result["worker_cpus"] = workers
            numa_maps = Path(f"/proc/{server.process.pid}/numa_maps").read_text()
            (directory / "manager-numa-maps.txt").write_text(numa_maps)
            pools = [line for line in numa_maps.splitlines() if "orbitkv-payload" in line]
            assert pools and any(re.search(rf"\bN{numa}=[1-9]\d*\b", line) for line in pools), pools
            if "," not in manager_visible:
                resident_nodes = {
                    int(node) for line in pools for node in re.findall(r"\bN(\d+)=", line)
                }
                assert resident_nodes == {numa}, pools
            result["payload_pools"] = pools
        worker_output.with_suffix(".release").touch()
        assert client.wait(timeout=60) == 0, (directory / "client.log").read_text()
        result["client_exit"] = client.returncode
        result["worker"] = json.loads(worker_output.read_text())
        assert result["worker"]["normal_unregister"]
        require_clean_identity_log((directory / "client.log").read_text())
        final_metrics = fetch_orbitkv_metrics(http_port)
        for gauge in (
            "orbitkv_query_reserved_bytes",
            "orbitkv_inflight_bytes",
            "orbitkv_transfer_lock_active",
            "orbitkv_transfer_reserved_bytes",
            "orbitkv_ssd_prefetch_inflight",
            "orbitkv_ssd_read_pinned_bytes",
            "orbitkv_ssd_write_queue_pending",
            "orbitkv_ssd_write_inflight",
        ):
            assert gauge in final_metrics and final_metrics[gauge] == 0, (gauge, final_metrics)
        result["final_metrics"] = final_metrics
        cleanup = drain_identity_processes(client, server, worker_output.with_suffix(".release"))
        assert not cleanup["errors"] and not cleanup["remaining_processes"], cleanup
        result["manager_exit"] = cleanup["processes"][1]["exit_code"]
        assert result["manager_exit"] == 0
        require_clean_identity_log(server.read_logs())
        assert not Path(server.bootstrap_socket).exists()
        for listener in (port, http_port):
            with socket.socket() as probe:
                probe.settimeout(1)
                assert probe.connect_ex(("127.0.0.1", listener)) != 0, listener
        completed = True
    except BaseException as error:
        values["failed"] = True
        result["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if cleanup is None:
            cleanup = drain_identity_processes(
                client, server, worker_output.with_suffix(".release")
            )
        result["cleanup"] = cleanup
        values["failed"] |= not completed or bool(cleanup["errors"])
        try:
            quarantine_identity_gpus(values["lock_root"], values["devices"], directory, cleanup)
        except OSError as error:
            values["failed"] = True
            cleanup["errors"].append(f"Quarantine write failed: {error}")
        result.update(
            completed=completed and not cleanup["errors"],
            client_exit=None if client is None else client.returncode,
        )
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if cleanup["errors"]:
            pytest.fail(
                f"Identity drain failed: {cleanup}; primary failure: {result.get('failure')}"
            )
