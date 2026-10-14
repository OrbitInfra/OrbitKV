"""Bounded owner-status polling shared by in-process and helper observers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

OBSERVER_PROTOCOL = "s2.10-observer-process-isolation-v1"
MISSED_DEADLINE_POLICY = "execute-every-slot-without-skip"
MAX_OWNER_RESPONSE_BYTES = 1_048_576
MAX_SAMPLE_BYTES = 65_536
PROTOCOL_POLL_SECONDS = 0.005


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _owner_status(endpoint: str, incarnation: str) -> tuple[dict | None, int]:
    response = requests.get(
        f"{endpoint}/cache/metadata/owners",
        params={"limit": 128},
        timeout=5,
    )
    response.raise_for_status()
    if len(response.content) > MAX_OWNER_RESPONSE_BYTES:
        raise RuntimeError(f"owner-status response exceeds {MAX_OWNER_RESPONSE_BYTES} bytes")
    rows = response.json()
    if not isinstance(rows, list):
        raise TypeError("owner-status response is not a list")
    return next((row for row in rows if row.get("owner") == incarnation), None), len(
        response.content
    )


def _wait_for(
    path: Path,
    stop: Path,
    timeout_seconds: float,
    cancel: threading.Event | None = None,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if (cancel is not None and cancel.is_set()) or (stop != path and stop.exists()):
            return False
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(PROTOCOL_POLL_SECONDS)
    return True


def _wait_until(
    target_ns: int,
    stop: Path,
    cancel: threading.Event | None = None,
) -> bool:
    while True:
        if (cancel is not None and cancel.is_set()) or stop.exists():
            return False
        remaining_ns = target_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return True
        time.sleep(min(remaining_ns / 1_000_000_000, PROTOCOL_POLL_SECONDS))


def _gpu_modules() -> list[str]:
    prefixes = ("torch", "orbitkv", "cupy", "cuda")
    return sorted(name for name in sys.modules if name.split(".", 1)[0] in prefixes)


def _run_observer(
    *,
    endpoint: str,
    incarnation: str,
    expected_view: str,
    output: Path,
    mode: str,
    expected_cadence_ns: int,
    max_polls: int,
    start_timeout_seconds: float,
    stop_timeout_seconds: float,
    cancel: threading.Event | None = None,
) -> dict:
    paths = {
        name: output / f"observer-{name}.json"
        for name in ("ready", "start", "complete", "stop", "exit", "error")
    }
    samples_path = output / "observer-samples.jsonl"
    if max_polls <= 0:
        raise ValueError("max_polls must be positive")
    if expected_cadence_ns <= 0:
        raise ValueError("expected_cadence_ns must be positive")
    if samples_path.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("observer protocol output already exists")
    loaded_modules = _gpu_modules()
    if mode == "helper-process" and loaded_modules:
        raise RuntimeError(f"helper loaded forbidden GPU/native modules: {loaded_modules}")

    _atomic_json(
        paths["ready"],
        {
            "contract": OBSERVER_PROTOCOL,
            "mode": mode,
            "pid": os.getpid(),
            "thread_id": threading.get_native_id(),
            "ready_mono_ns": time.monotonic_ns(),
            "python_executable": sys.executable,
            "python_path": sys.path,
            "requests_origin": requests.__file__,
            "gpu_or_native_modules_loaded": loaded_modules,
            "max_polls": max_polls,
            "expected_cadence_ns": expected_cadence_ns,
        },
    )
    if not _wait_for(paths["start"], paths["stop"], start_timeout_seconds, cancel):
        result = {
            "contract": OBSERVER_PROTOCOL,
            "mode": mode,
            "status": "stopped_before_start",
            "samples": 0,
            "pid": os.getpid(),
        }
        _atomic_json(paths["exit"], result)
        return result

    start = _read_json(paths["start"])
    required = {
        "contract": OBSERVER_PROTOCOL,
        "cadence_ns": expected_cadence_ns,
        "poll_count": max_polls,
        "missed_deadline_policy": MISSED_DEADLINE_POLICY,
    }
    for name, expected in required.items():
        if start.get(name) != expected:
            raise ValueError(f"observer start {name}={start.get(name)!r}, expected {expected!r}")
    epoch_ns = start["epoch_mono_ns"]
    phase_offset_ns = start["observer_phase_offset_ns"]
    if not isinstance(epoch_ns, int) or epoch_ns <= 0:
        raise ValueError("observer epoch must be a positive integer")
    if not isinstance(phase_offset_ns, int) or phase_offset_ns < 0:
        raise ValueError("observer phase offset must be a nonnegative integer")
    first_scheduled_ns = epoch_ns + phase_offset_ns
    total_cpu_ns = 0
    response_bytes = 0
    samples_digest = hashlib.sha256()
    first_actual_ns = None
    last_actual_ns = None
    max_lateness_ns = 0
    with samples_path.open("x", buffering=1) as samples_file:
        for index in range(max_polls):
            scheduled_ns = first_scheduled_ns + index * expected_cadence_ns
            if not _wait_until(scheduled_ns, paths["stop"], cancel):
                raise RuntimeError(f"observer stopped after {index}/{max_polls} polls")
            poll_start_ns = time.monotonic_ns()
            cpu_start_ns = time.thread_time_ns()
            owner, received_bytes = _owner_status(endpoint, incarnation)
            poll_end_ns = time.monotonic_ns()
            cpu_ns = time.thread_time_ns() - cpu_start_ns
            if not owner or not owner.get("fresh") or owner.get("view_id") != expected_view:
                raise RuntimeError(f"invalid owner status at poll {index}: {owner!r}")
            row = {
                "index": index,
                "scheduled_poll_start_mono_ns": scheduled_ns,
                "poll_start_mono_ns": poll_start_ns,
                "poll_end_mono_ns": poll_end_ns,
                "scheduling_deviation_ns": poll_start_ns - scheduled_ns,
                "sampler_cpu_ns": cpu_ns,
                "response_bytes": received_bytes,
                "owner": owner,
            }
            encoded = (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode()
            if len(encoded) > MAX_SAMPLE_BYTES:
                raise RuntimeError(f"observer sample exceeds {MAX_SAMPLE_BYTES} bytes")
            samples_file.write(encoded.decode())
            samples_digest.update(encoded)
            total_cpu_ns += cpu_ns
            response_bytes += received_bytes
            first_actual_ns = first_actual_ns or poll_start_ns
            last_actual_ns = poll_start_ns
            max_lateness_ns = max(max_lateness_ns, poll_start_ns - scheduled_ns)

    complete = {
        "contract": OBSERVER_PROTOCOL,
        "mode": mode,
        "status": "complete",
        "pid": os.getpid(),
        "samples": max_polls,
        "samples_sha256": samples_digest.hexdigest(),
        "response_bytes": response_bytes,
        "sampler_cpu_ns": total_cpu_ns,
        "first_scheduled_mono_ns": first_scheduled_ns,
        "last_scheduled_mono_ns": first_scheduled_ns + (max_polls - 1) * expected_cadence_ns,
        "first_actual_mono_ns": first_actual_ns,
        "last_actual_mono_ns": last_actual_ns,
        "max_scheduling_deviation_ns": max_lateness_ns,
    }
    _atomic_json(paths["complete"], complete)
    if not _wait_for(paths["stop"], paths["stop"], stop_timeout_seconds, cancel):
        raise AssertionError("unreachable observer stop wait")
    result = {**complete, "status": "exited", "exit_mono_ns": time.monotonic_ns()}
    _atomic_json(paths["exit"], result)
    return result


def _record_error(output: Path, mode: str, error: BaseException) -> None:
    path = output / "observer-error.json"
    if path.exists():
        return
    record = {
        "contract": OBSERVER_PROTOCOL,
        "mode": mode,
        "status": "error",
        "pid": os.getpid(),
        "error_type": type(error).__name__,
        "error": repr(error),
        "error_mono_ns": time.monotonic_ns(),
    }
    _atomic_json(path, record)
    exit_path = output / "observer-exit.json"
    if not exit_path.exists():
        _atomic_json(exit_path, record)


@dataclass
class ObserverSession:
    mode: str
    endpoint: str
    incarnation: str
    expected_view: str
    output: Path
    cadence_ns: int
    poll_count: int
    process: subprocess.Popen | None = field(default=None, init=False)
    thread: threading.Thread | None = field(default=None, init=False)
    thread_errors: list[BaseException] = field(default_factory=list, init=False)
    cleanup_errors: list[BaseException] = field(default_factory=list, init=False)
    thread_cancel: threading.Event = field(default_factory=threading.Event, init=False)
    started: bool = field(default=False, init=False)
    finished: bool = field(default=False, init=False)
    stop_attempted: bool = field(default=False, init=False)
    forced_cleanup: str | None = field(default=None, init=False)

    def _path(self, name: str) -> Path:
        return self.output / f"observer-{name}.json"

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def launch(self, timeout_seconds: float = 30) -> dict:
        if self.mode not in {"in-process", "helper-process"}:
            raise ValueError(f"unsupported observer mode: {self.mode}")
        try:
            if self.mode == "in-process":

                def target() -> None:
                    try:
                        _run_observer(
                            endpoint=self.endpoint,
                            incarnation=self.incarnation,
                            expected_view=self.expected_view,
                            output=self.output,
                            mode=self.mode,
                            expected_cadence_ns=self.cadence_ns,
                            max_polls=self.poll_count,
                            start_timeout_seconds=timeout_seconds,
                            stop_timeout_seconds=timeout_seconds,
                            cancel=self.thread_cancel,
                        )
                    except BaseException as error:
                        self.thread_errors.append(error)
                        _record_error(self.output, self.mode, error)

                self.thread = threading.Thread(
                    target=target,
                    name="inventory-observer-sampler",
                    daemon=True,
                )
                self.thread.start()
            else:
                log = (self.output / "observer-helper.log").open("xb")
                command = [
                    sys.executable,
                    "-m",
                    "benches.observer_polling",
                    "--helper",
                    "--endpoint",
                    self.endpoint,
                    "--incarnation",
                    self.incarnation,
                    "--expected-view",
                    self.expected_view,
                    "--output",
                    str(self.output),
                    "--cadence-ns",
                    str(self.cadence_ns),
                    "--max-polls",
                    str(self.poll_count),
                    "--start-timeout-seconds",
                    str(timeout_seconds),
                    "--stop-timeout-seconds",
                    str(timeout_seconds),
                ]
                try:
                    self.process = subprocess.Popen(
                        command,
                        cwd=Path(__file__).resolve().parents[1],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                finally:
                    log.close()
            deadline = time.monotonic() + timeout_seconds
            while not self._path("ready").exists():
                self.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("observer did not become ready")
                time.sleep(PROTOCOL_POLL_SECONDS)
            ready = _read_json(self._path("ready"))
            if ready["contract"] != OBSERVER_PROTOCOL or ready["mode"] != self.mode:
                raise RuntimeError(f"invalid observer ready record: {ready}")
            if self.mode == "helper-process" and ready["gpu_or_native_modules_loaded"]:
                raise RuntimeError(f"helper imported forbidden modules: {ready}")
            return ready
        except BaseException:
            self.abort()
            raise

    def start(
        self,
        *,
        epoch_mono_ns: int,
        observer_phase_offset_ns: int,
        foreground_phase_offset_ns: int,
    ) -> dict:
        start = {
            "contract": OBSERVER_PROTOCOL,
            "token": hashlib.sha256(
                f"{epoch_mono_ns}:{self.incarnation}:{self.expected_view}".encode()
            ).hexdigest(),
            "epoch_mono_ns": epoch_mono_ns,
            "observer_phase_offset_ns": observer_phase_offset_ns,
            "foreground_phase_offset_ns": foreground_phase_offset_ns,
            "cadence_ns": self.cadence_ns,
            "poll_count": self.poll_count,
            "missed_deadline_policy": MISSED_DEADLINE_POLICY,
        }
        _atomic_json(self._path("start"), start)
        self.started = True
        return start

    def check(self) -> None:
        error_path = self._path("error")
        if error_path.exists():
            raise RuntimeError(f"observer failed: {_read_json(error_path)}")
        if self.thread_errors:
            raise RuntimeError(f"observer failed: {self.thread_errors!r}")
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"observer helper exited early with {self.process.returncode}")
        if (
            self.thread is not None
            and not self.thread.is_alive()
            and not self._path("exit").exists()
        ):
            raise RuntimeError("in-process observer exited without protocol record")

    def finish(self, timeout_seconds: float = 30) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while not self._path("complete").exists():
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("observer did not complete its bounded polls")
            time.sleep(PROTOCOL_POLL_SECONDS)
        _atomic_json(
            self._path("stop"),
            {
                "contract": OBSERVER_PROTOCOL,
                "requested_mono_ns": time.monotonic_ns(),
                "reason": "bounded_poll_count_complete",
            },
        )
        while not self._path("exit").exists():
            self.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("observer did not acknowledge stop")
            time.sleep(PROTOCOL_POLL_SECONDS)
        if self.thread is not None:
            self.thread.join(timeout=1)
            if self.thread.is_alive():
                raise RuntimeError("in-process observer remained alive after exit")
        if self.process is not None:
            exit_code = self.process.wait(timeout=1)
            if exit_code != 0:
                raise RuntimeError(f"observer helper exited with {exit_code}")
        result = _read_json(self._path("exit"))
        if result["status"] != "exited" or result["samples"] != self.poll_count:
            raise RuntimeError(f"incomplete observer exit: {result}")
        sample_bytes = (self.output / "observer-samples.jsonl").read_bytes()
        rows = [json.loads(line) for line in sample_bytes.splitlines()]
        if len(rows) != self.poll_count:
            raise RuntimeError(f"observer wrote {len(rows)}/{self.poll_count} samples")
        if hashlib.sha256(sample_bytes).hexdigest() != result["samples_sha256"]:
            raise RuntimeError("observer sample digest mismatch")
        for index, row in enumerate(rows):
            expected = result["first_scheduled_mono_ns"] + index * self.cadence_ns
            if row["index"] != index or row["scheduled_poll_start_mono_ns"] != expected:
                raise RuntimeError(f"observer schedule mismatch at sample {index}")
            if (
                row["poll_start_mono_ns"] < expected
                or row["poll_end_mono_ns"] < row["poll_start_mono_ns"]
            ):
                raise RuntimeError(f"invalid observer timestamps at sample {index}")
        self.finished = True
        return result

    def abort(self) -> None:
        if self.finished:
            return
        self.thread_cancel.set()
        stop_failed = False
        if not self.stop_attempted:
            self.stop_attempted = True
            try:
                if not self._path("stop").exists():
                    _atomic_json(
                        self._path("stop"),
                        {
                            "contract": OBSERVER_PROTOCOL,
                            "requested_mono_ns": time.monotonic_ns(),
                            "reason": "cell_cleanup",
                        },
                    )
            except BaseException as error:
                self.cleanup_errors.append(error)
                stop_failed = True
        if self.thread is not None:
            try:
                self.thread.join(timeout=6)
                if self.thread.is_alive():
                    self.cleanup_errors.append(
                        RuntimeError("in-process observer remained alive during cleanup")
                    )
            except BaseException as error:
                self.cleanup_errors.append(error)
        if self.process is not None:
            try:
                if stop_failed and self.process.poll() is None:
                    self.forced_cleanup = "terminate"
                    self.process.terminate()
                if self.process.poll() is None:
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.forced_cleanup = "terminate"
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.forced_cleanup = "kill"
                            self.process.kill()
                            self.process.wait(timeout=5)
            except BaseException as error:
                self.cleanup_errors.append(error)
                try:
                    if self.process.poll() is None:
                        self.forced_cleanup = "kill"
                        self.process.kill()
                        self.process.wait(timeout=5)
                except BaseException as fallback_error:
                    self.cleanup_errors.append(fallback_error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", action="store_true", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--incarnation", required=True)
    parser.add_argument("--expected-view", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cadence-ns", type=int, required=True)
    parser.add_argument("--max-polls", type=int, required=True)
    parser.add_argument("--start-timeout-seconds", type=float, default=30)
    parser.add_argument("--stop-timeout-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        _run_observer(
            endpoint=args.endpoint,
            incarnation=args.incarnation,
            expected_view=args.expected_view,
            output=args.output,
            mode="helper-process",
            expected_cadence_ns=args.cadence_ns,
            max_polls=args.max_polls,
            start_timeout_seconds=args.start_timeout_seconds,
            stop_timeout_seconds=args.stop_timeout_seconds,
        )
    except BaseException as error:
        _record_error(args.output, "helper-process", error)
        raise


if __name__ == "__main__":
    main()
