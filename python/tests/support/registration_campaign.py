"""Execute one frozen installed registration gate with durable resource ownership."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(pid: int) -> dict:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "parent": int(fields[1]),
        "group": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def descendants() -> list[dict]:
    table = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            item = identity(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        table[item["pid"]] = item
    if os.getpid() not in table:
        raise RuntimeError("Controller is outside its /proc namespace")
    owned = {os.getpid()}
    while True:
        larger = owned | {pid for pid, item in table.items() if item["parent"] in owned}
        if larger == owned:
            return [table[pid] for pid in sorted(owned - {os.getpid()})]
        owned = larger


def inventory(gpu_uuid: str) -> dict:
    gpu = (
        subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu_uuid,
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        .stdout.strip()
        .split(",")
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout
    if len(gpu) != 2 or gpu[0].strip() != gpu_uuid:
        raise RuntimeError("GPU inventory does not match the frozen UUID")
    return {
        "uuid": gpu_uuid,
        "memory_mib": int(gpu[1]),
        "compute": [row for row in apps.splitlines() if gpu_uuid in row],
    }


def check_inputs(contract: dict) -> None:
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != contract["boot_id"]:
        raise RuntimeError("Host boot identity changed")
    for path, expected in contract["input_sha256"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    protected = contract["protected_gpu0"]
    if (
        digest(Path(protected["quarantine"])) != protected["quarantine_sha256"]
        or Path(protected["lease"]).stat().st_ino != protected["lease_inode"]
        or identity(protected["guardian_pid"])["start_ticks"] != protected["guardian_start_ticks"]
    ):
        raise RuntimeError("Protected resource identity changed")


def quarantine(root: Path, reason: str, locks: list[int]) -> None:
    with contextlib.suppress(BaseException):
        write(
            root / "QUARANTINE.json",
            {
                "reason": reason,
                "controller": identity(os.getpid()),
                "locks": locks,
                "physical_termination_claimed": False,
            },
        )
    while True:
        with contextlib.suppress(BaseException):
            time.sleep(1)


def validate_cell(cell: dict, directory: Path, pytest_owner: dict) -> None:
    cases = list(ET.parse(directory / "junit.xml").getroot().iter("testcase"))
    if len(cases) != cell["passed"] or any(
        case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")
    ):
        raise RuntimeError("Frozen pytest outcome differs")
    receipts = list(directory.rglob("*.cleanup.json"))
    if len(receipts) != cell["cleanup_receipts"]:
        raise RuntimeError("Missing native service cleanup receipts")
    for path in receipts:
        row = json.loads(path.read_text())
        if (
            type(row.get("pid")) is not int
            or type(row.get("start_ticks")) is not int
            or row.get("forced_kill") is not False
            or row.get("reaped") is not True
            or type(row.get("exit_code")) is not int
            or row.get("exit_code") not in cell["normal_exit_codes"]
            or row.get("remaining_processes") != []
            or row.get("inspection_errors") != []
        ):
            raise RuntimeError(f"Native cleanup is not established: {path}")
    for path in directory.rglob("*.log"):
        text = path.read_text(errors="replace")
        if (
            "Traceback (most recent call last)" in text
            or "EngineDeadError" in text
            or "panicked at" in text
        ):
            raise RuntimeError(f"Native/worker failure remains in log: {path}")
    if cell["kind"] == "native":
        releases = list(directory.rglob("*.native-release.json"))
        if len(releases) != 4:
            raise RuntimeError("Missing complete native client release receipts")
        for path in releases:
            row = json.loads(path.read_text())
            if (
                row.get("pid") != pytest_owner["pid"]
                or row.get("start_ticks") != pytest_owner["start_ticks"]
                or row.get("closed") is not True
            ):
                raise RuntimeError("Native client release identity differs")
    if cell["kind"] == "sglang-pool":
        releases = list(directory.rglob("sglang-registration.native-release.json"))
        if len(releases) != 2:
            raise RuntimeError("Missing SGLang pool lifetime receipts")
        for path in releases:
            row = json.loads(path.read_text())
            if (
                row.get("pid") != pytest_owner["pid"]
                or row.get("start_ticks") != pytest_owner["start_ticks"]
                or row.get("workers_joined") is not True
                or row.get("exporters_after_last_hold") != 0
                or row.get("offload_completed") is not True
                or row.get("registration_retired") is not True
                or row.get("client_closed") is not True
            ):
                raise RuntimeError("SGLang pool lifetime contract differs")
    if cell["kind"] == "credits":
        credits = list(directory.rglob("*.credit-state.json"))
        if len(credits) != 2:
            raise RuntimeError("Missing explicit credit-state receipts")
        for path in credits:
            row = json.loads(path.read_text())
            if (
                row.get("pid") != pytest_owner["pid"]
                or row.get("start_ticks") != pytest_owner["start_ticks"]
                or row.get("credit_counter_exists_and_zero") is not True
                or row.get("physical_fault_drain_claimed") is not False
            ):
                raise RuntimeError("Credit receipt identity or scope differs")
        partial = json.loads(next(directory.rglob("partial-export.credit-state.json")).read_text())
        normal = json.loads(next(directory.rglob("duplicate-export.credit-state.json")).read_text())
        if (
            partial.get("framework_throw_retained") is not True
            or partial.get("unregister_refused") is not True
            or partial.get("native_registration_or_payload_submitted") is not False
            or partial.get("native_retained_attempts", 0) < 1
            or normal.get("export_calls") != 1
            or normal.get("duplicate_refused_before_export") is not True
            or normal.get("unregister_succeeded") is not True
        ):
            raise RuntimeError("Credit repayment/retention contract failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()
    if not __debug__ or os.environ.get("PYTHONOPTIMIZE"):
        raise RuntimeError("Optimized controller execution is forbidden")
    contract_bytes = args.contract.read_bytes()
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    contract = json.loads(contract_bytes)
    root = args.contract.parent
    if contract["phase"] not in ("precheck", "formal"):
        raise RuntimeError("Unknown registration gate phase")
    if contract["schema"] != "orbitkv.explicit-registration-campaign.v1":
        raise RuntimeError("Unknown campaign schema")
    expected = (
        [("native", 2)]
        if contract["phase"] == "precheck"
        else [("credits", 2), ("native", 2), ("vllm-worker", 1), ("sglang-pool", 2)]
    )
    if [(cell["kind"], cell["passed"]) for cell in contract["cells"]] != expected:
        raise RuntimeError("Frozen registration gate cells differ")
    for field, maximum in (
        ("review_wait_seconds", 3600),
        ("campaign_seconds", 7200),
        ("cleanup_seconds", 180),
    ):
        if type(contract[field]) is not int or not 0 < contract[field] <= maximum:
            raise RuntimeError("Invalid bounded campaign/review/cleanup budget")
    if any(type(cell["passed"]) is not int for cell in contract["cells"]):
        raise RuntimeError("Invalid declared gate count")
    if identity(os.getpid())["pid"] != os.getpid():
        raise RuntimeError("Invalid controller process identity")
    locks = []
    active = False
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError("Cannot own orphaned native workers")
    for number in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, signal.SIG_IGN)
    try:
        check_inputs(contract)
        cpus = contract["cpus"]
        if not cpus or not set(cpus) <= os.sched_getaffinity(0):
            raise RuntimeError("Frozen CPU affinity is unavailable")
        os.sched_setaffinity(0, cpus)
        for resource in contract["locks"]:
            fd = os.open(resource["path"], os.O_RDWR | os.O_NOFOLLOW)
            locks.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if os.fstat(fd).st_ino != resource["inode"]:
                raise RuntimeError("Resource lock inode changed")
        if Path(contract["quarantine"]).exists():
            raise RuntimeError("GPU resource is quarantined")
        before = inventory(contract["gpu_uuid"])
        if before["memory_mib"] != 0 or before["compute"]:
            raise RuntimeError("GPU resource is in use")
        write(
            root / "READY.json",
            {
                "controller": identity(os.getpid()),
                "contract_sha256": digest(args.contract),
                "locks": [
                    {"path": row["path"], "inode": os.fstat(fd).st_ino}
                    for row, fd in zip(contract["locks"], locks, strict=True)
                ],
                "inventory": before,
                "GPU_launches": 0,
                "phase": contract["phase"],
            },
        )
        deadline = time.monotonic() + contract["review_wait_seconds"]
        approval = root / "LAUNCH-APPROVED.json"
        while not approval.exists():
            if time.monotonic() >= deadline:
                write(
                    root / "NOT-LAUNCHED.json", {"GPU_launches": 0, "reason": "review wait expired"}
                )
                return
            time.sleep(0.25)
        if approval.is_symlink() or args.contract.read_bytes() != contract_bytes:
            raise RuntimeError("Launch grant or frozen contract changed")
        grant = json.loads(approval.read_text())
        if (
            grant["approved"] is not True
            or grant["phase"] != contract["phase"]
            or grant["contract_sha256"] != contract_sha
            or len(grant["review_sha256"]) != 64
        ):
            raise RuntimeError("Independent launch grant differs")
        global_deadline = time.monotonic() + contract["campaign_seconds"]
        cleanup_spent = 0.0
        completed = []
        for cell in contract["cells"]:
            if time.monotonic() >= global_deadline or args.contract.read_bytes() != contract_bytes:
                raise RuntimeError("Campaign budget exhausted or contract changed before admission")
            check_inputs(contract)
            if time.monotonic() >= global_deadline:
                raise RuntimeError("Campaign budget exhausted by input verification")
            directory = root / cell["name"]
            directory.mkdir()
            env = {
                key: value
                for key, value in os.environ.items()
                if key.lower() not in ("http_proxy", "https_proxy", "all_proxy")
                and key
                not in (
                    "PYTHONPATH",
                    "PYTHONHOME",
                    "PYTHONOPTIMIZE",
                    "CUDA_VISIBLE_DEVICES",
                    "LD_PRELOAD",
                    "LD_LIBRARY_PATH",
                )
                and not key.startswith(("ORBITKV_", "MC_", "TENT_"))
            }
            env.update(cell["environment"])
            if (
                env.get("CUDA_VISIBLE_DEVICES") != contract["gpu_uuid"]
                or env.get("ORBITKV_TEST_STRICT_SHUTDOWN") != "1"
                or env.get("MC_FORCE_TCP") != "1"
                or env.get("LD_PRELOAD")
            ):
                raise RuntimeError("Frozen installed TCP/strict/GPU environment is invalid")
            env["ORBITKV_TEST_GPU_LOCK_FDS"] = json.dumps(locks)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            with (directory / "pytest.log").open("x") as log:
                active = True
                process = subprocess.Popen(
                    cell["argv"],
                    cwd=cell["cwd"],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=tuple(locks),
                )
                owner = identity(process.pid)
                write(directory / "STARTED.json", owner)
                deadline = min(global_deadline, time.monotonic() + cell["seconds"])
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        quarantine(
                            root, "Native process deadline without terminal ownership", locks
                        )
                    time.sleep(0.1)
                code = process.wait()
                write(directory / "WAIT.json", dict(owner, exit_code=code, actual_wait=True))
            cleanup_started = time.monotonic()
            cleanup_deadline = cleanup_started + max(
                0.0, contract["cleanup_seconds"] - cleanup_spent
            )
            while descendants():
                for child in descendants():
                    if child["parent"] == os.getpid():
                        pid, status = os.waitpid(child["pid"], os.WNOHANG)
                        if pid:
                            write(
                                directory / f"ADOPTED-{pid}.json",
                                dict(
                                    child,
                                    exit_code=os.waitstatus_to_exitcode(status),
                                    actual_wait=True,
                                ),
                            )
                if time.monotonic() >= cleanup_deadline:
                    quarantine(
                        root, "Native descendants remain; cumulative cleanup exhausted", locks
                    )
                time.sleep(0.1)
            after = inventory(contract["gpu_uuid"])
            if after["memory_mib"] != 0 or after["compute"]:
                quarantine(root, "GPU occupancy remains after owned-process waits", locks)
            validate_cell(cell, directory, owner)
            cleanup_spent += time.monotonic() - cleanup_started
            if cleanup_spent > contract["cleanup_seconds"]:
                quarantine(root, "Cumulative cleanup/oracle budget exceeded", locks)
            active = False
            if code != 0:
                raise RuntimeError("First invalid formal cell; no retries or replacement")
            completed.append(cell["name"])
            write(directory / "COMPLETE.json", {"completed": True, "inventory": after})
        check_inputs(contract)
        if time.monotonic() >= global_deadline:
            raise RuntimeError("Campaign budget exhausted before final acceptance")
        write(
            root / "RESULT.json",
            {
                "state": "completed_pending_review",
                "cells": completed,
                "physical_fault_qualification": False,
                "forced_cleanup": False,
                "phase": contract["phase"],
                "formal_cells": len(completed) if contract["phase"] == "formal" else 0,
            },
        )
    except BaseException as error:
        if active:
            quarantine(root, repr(error), locks)
        write(root / "FAILURE.json", {"state": "fail_stop", "error": repr(error)})
        raise
    finally:
        for fd in locks:
            os.close(fd)


if __name__ == "__main__":
    main()
