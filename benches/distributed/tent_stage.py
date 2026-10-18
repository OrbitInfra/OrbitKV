"""Split native TENT READ startup with private CPU buffers and frozen libraries."""

from __future__ import annotations

import argparse
import contextlib
import ctypes as c
import hashlib
import json
import mmap
import os
import signal
import sys
import time
from pathlib import Path


class MemoryOptions(c.Structure):
    _fields_ = [
        ("location", c.c_char * 64),
        ("permission", c.c_int),
        ("transport", c.c_int),
        ("shm_path", c.c_char * 256),
        ("shm_offset", c.c_size_t),
        ("internal", c.c_int),
    ]


class Request(c.Structure):
    _fields_ = [
        ("opcode", c.c_int),
        ("source", c.c_void_p),
        ("target_id", c.c_uint64),
        ("target_offset", c.c_uint64),
        ("length", c.c_uint64),
        ("priority", c.c_int),
        ("hint", c.c_int),
    ]


class Status(c.Structure):
    _fields_ = [("status", c.c_int), ("transferred_bytes", c.c_uint64)]


class Notification(c.Structure):
    _fields_ = [("handle", c.c_uint64), ("name", c.c_char * 256), ("msg", c.c_char * 4096)]


class Notifications(c.Structure):
    _fields_ = [("num_records", c.c_int), ("records", c.POINTER(Notification))]


def emit(value: dict) -> None:
    print(json.dumps({"tent_stage_probe": True, **value}), flush=True)


def checked(value: int, operation: str) -> None:
    if value != 0:
        raise RuntimeError(f"{operation}: {value}")


def drain_batch(api, engine, batch: int, failure: str | None, timeout: float) -> dict:
    """An error or timeout requests cancellation; it never releases live memory."""
    started = time.perf_counter_ns()
    polls = 0
    cancelled = False
    terminal = None
    freed = False
    while True:
        try:
            if terminal is None:
                status = Status()
                rc = api.tent_task_status(engine, batch, 0, c.byref(status))
                polls += 1
                if rc:
                    failure = failure or f"poll: {rc}"
                elif status.status in (2, 3, 4, 5, 6):
                    terminal = status
                    terminal_ns = time.perf_counter_ns()
                    if status.status != 4:
                        failure = failure or f"terminal: {status.status}"
                elif status.status not in (0, 1):
                    failure = failure or f"unknown status: {status.status}"
            if terminal is not None:
                if api.tent_free_batch(engine, batch) == 0:
                    freed = True
                    freed_ns = time.perf_counter_ns()
                    if failure:
                        raise RuntimeError(failure)
                    return {
                        "poll_to_terminal_ms": (terminal_ns - started) / 1e6,
                        "free_ms": (freed_ns - terminal_ns) / 1e6,
                        "terminal_mono_ns": terminal_ns,
                        "freed_mono_ns": freed_ns,
                        "poll_calls": polls,
                        "transferred_bytes": terminal.transferred_bytes,
                    }
            elif not cancelled and (failure or (time.perf_counter_ns() - started) / 1e9 >= timeout):
                failure = failure or "transfer deadline exceeded"
                cancelled = True
                checked(api.tent_cancel_task(engine, batch, 0), "cancel")
        except BaseException as error:
            if freed:
                raise
            failure = failure or repr(error)
        if failure or time.perf_counter_ns() - started >= 1_000_000:
            time.sleep(0.001)


def load_api(directory: Path):
    libraries = [
        c.CDLL(str(directory / name), mode=c.RTLD_GLOBAL)
        for name in ("libasio.so", "libmooncake_common.so", "libtent_shared.so")
    ]
    api = libraries[-1]
    signatures = {
        "tent_set_config": ([c.c_char_p, c.c_char_p], None),
        "tent_create_engine": ([], c.c_void_p),
        "tent_available": ([c.c_void_p], c.c_int),
        "tent_destroy_engine": ([c.c_void_p], None),
        "tent_segment_name": ([c.c_void_p, c.c_char_p, c.c_size_t], c.c_int),
        "tent_open_segment": ([c.c_void_p, c.POINTER(c.c_uint64), c.c_char_p], c.c_int),
        "tent_close_segment": ([c.c_void_p, c.c_uint64], c.c_int),
        "tent_register_memory_ex": (
            [c.c_void_p, c.c_void_p, c.c_size_t, c.POINTER(MemoryOptions)],
            c.c_int,
        ),
        "tent_unregister_memory": ([c.c_void_p, c.c_void_p, c.c_size_t], c.c_int),
        "tent_allocate_batch": ([c.c_void_p, c.c_size_t], c.c_uint64),
        "tent_submit": ([c.c_void_p, c.c_uint64, c.POINTER(Request), c.c_size_t], c.c_int),
        "tent_task_status": ([c.c_void_p, c.c_uint64, c.c_size_t, c.POINTER(Status)], c.c_int),
        "tent_cancel_task": ([c.c_void_p, c.c_uint64, c.c_size_t], c.c_int),
        "tent_free_batch": ([c.c_void_p, c.c_uint64], c.c_int),
        "tent_send_notifs": ([c.c_void_p, c.c_uint64, c.c_char_p, c.c_char_p], c.c_int),
        "tent_recv_notifs": ([c.c_void_p, c.POINTER(Notifications)], c.c_int),
        "tent_free_notifs": ([c.POINTER(Notifications)], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes = arguments
        function.restype = result
    return libraries, api


def notification_messages(run_id: str, count: int) -> dict[bytes, bytes]:
    if not run_id.isascii() or not 1 <= len(run_id) <= 64 or not 1 <= count <= 1024:
        raise ValueError("Notifications require an ASCII run id and 1..1024 messages")
    return {
        f"{run_id}/{index:04d}".encode(): hashlib.shake_256(f"{run_id}:{index}".encode())
        .hexdigest(1024)
        .encode()
        for index in range(count)
    }


def receive_notifications(api, engine, expected: dict[bytes, bytes], timeout: float) -> dict:
    remaining = dict(expected)
    deadline = time.monotonic() + timeout
    quiet_until = None
    polls = 0
    while True:
        info = Notifications()
        try:
            checked(api.tent_recv_notifs(engine, c.byref(info)), "receive notifications")
            polls += 1
            if not 0 <= info.num_records <= len(expected) or (
                info.num_records and not info.records
            ):
                raise RuntimeError("Invalid native notification records")
            for index in range(info.num_records):
                record = info.records[index]
                if remaining.pop(record.name, None) != record.msg:
                    raise RuntimeError("Unexpected, duplicate or corrupt notification")
        finally:
            api.tent_free_notifs(c.byref(info))
        now = time.monotonic()
        if not remaining:
            quiet_until = quiet_until or now + 0.025
            if now >= quiet_until:
                return {"received": len(expected), "polls": polls, "quiet_ms": 25}
        if now >= deadline:
            raise RuntimeError(f"Notification deadline: {len(remaining)} missing")
        time.sleep(0.001)


def wait_source_stop() -> None:
    with contextlib.suppress(OSError):
        emit({"event": "source_control_error", "ownership_held": True})
    while True:
        try:
            line = sys.stdin.readline()
            if line and json.loads(line).get("operation") == "stop":
                return
        except (ValueError, OSError):
            pass
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--native-lib", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("source", "consumer"))
    parser.add_argument("--bytes", type=int, default=8 * 1024**2)
    args = parser.parse_args()
    if args.bytes <= 0:
        parser.error("--bytes must be positive")
    libraries, api = load_api(args.native_lib)
    for name, value in {
        "metadata_type": "p2p",
        "metadata_servers": "P2PHANDSHAKE",
        "local_segment_name": args.bind + ":0",
        "rpc_server_hostname": args.bind,
        "rpc_server_port": "0",
        "metrics/enabled": "false",
        "use_legacy_transport_selection": "false",
        "enable_runtime_queue": "false",
        "enable_auto_failover_on_poll": "true",
        "transports/rdma/enable": "true",
        **{
            f"transports/{name}/enable": "false"
            for name in ("tcp", "shm", "nvlink", "mnnvl", "gds", "io_uring")
        },
    }.items():
        api.tent_set_config(name.encode(), value.encode())
    # Control loss cannot revoke a buffer that a remote reader may still use.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: None)
    pattern = hashlib.shake_256(b"tent-cold-read-stage-20261010").digest(args.bytes)
    expected = hashlib.sha256(pattern).hexdigest()
    buffer = mmap.mmap(-1, args.bytes)
    buffer[:] = pattern
    owner = c.c_ubyte.from_buffer(buffer)
    address = c.addressof(owner)
    engine = None
    registered = False
    source_revoked = False
    segment_ids = set()
    try:
        started = time.perf_counter_ns()
        engine = api.tent_create_engine()
        if not engine or api.tent_available(engine) != 1:
            raise RuntimeError("TENT engine unavailable")
        initialized = time.perf_counter_ns()
        options = MemoryOptions(location=b"*", permission=2, transport=1)
        checked(
            api.tent_register_memory_ex(engine, address, args.bytes, c.byref(options)), "register"
        )
        registered = True
        registration_end = time.perf_counter_ns()
        endpoint = c.create_string_buffer(256)
        checked(api.tent_segment_name(engine, endpoint, len(endpoint)), "endpoint")
        emit(
            {
                "event": "ready",
                "endpoint": endpoint.value.decode(),
                "pid": os.getpid(),
                "address": address,
                "bytes": args.bytes,
                "sha256": expected,
                "initialize_ms": (initialized - started) / 1e6,
                "registration_ms": (registration_end - initialized) / 1e6,
                "memory": "Private CPU mmap; not Manager pinned pool or GPU memory",
            }
        )
        while True:
            line = sys.stdin.readline()
            if not line:
                emit({"event": "control_lost", "ownership_held": True})
                while True:
                    time.sleep(1)
            message = json.loads(line)
            if message["operation"] == "stop":
                source_revoked = True
                break
            if message["operation"] in ("send_notifications", "receive_notifications"):
                notifications = notification_messages(message["id"], message["count"])
                if message["operation"] == "receive_notifications":
                    emit({"event": "receiving_notifications", "id": message["id"]})
                    received = receive_notifications(api, engine, notifications, timeout=30)
                    emit({"event": "notifications_received", "id": message["id"], **received})
                else:
                    segment = c.c_uint64()
                    checked(
                        api.tent_open_segment(
                            engine, c.byref(segment), message["endpoint"].encode()
                        ),
                        "open notification segment",
                    )
                    segment_ids.add(segment.value)
                    for name, payload in notifications.items():
                        checked(
                            api.tent_send_notifs(engine, segment.value, name, payload),
                            "send notification",
                        )
                    emit(
                        {
                            "event": "notifications_sent",
                            "id": message["id"],
                            "sent": len(notifications),
                        }
                    )
                continue
            if args.role != "consumer":
                raise ValueError("Source accepts only stop after every reader has drained")
            if message["operation"] != "read" or not 1 <= message["iterations"] <= 1000:
                raise ValueError("Expected read with 1..1000 iterations or stop")
            samples = []
            for index in range(message["iterations"]):
                buffer[:] = b"\xa5" * args.bytes
                segment = c.c_uint64()
                begin = time.perf_counter_ns()
                checked(
                    api.tent_open_segment(engine, c.byref(segment), message["endpoint"].encode()),
                    "open",
                )
                opened = time.perf_counter_ns()
                segment_ids.add(segment.value)
                batch = api.tent_allocate_batch(engine, 1)
                if batch == 0:
                    raise RuntimeError("Batch allocation failed")
                allocated = time.perf_counter_ns()
                request = Request(0, address, segment.value, message["address"], args.bytes, 0, 1)
                failure = None
                try:
                    rc = api.tent_submit(engine, batch, c.byref(request), 1)
                    if rc:
                        failure = f"submit: {rc}"
                except BaseException as error:
                    failure = repr(error)
                submit_end = time.perf_counter_ns()
                drained = drain_batch(api, engine, batch, failure, timeout=30)
                freed = time.perf_counter_ns()
                actual = hashlib.sha256(buffer).hexdigest()
                if actual != expected or drained["transferred_bytes"] != args.bytes:
                    raise RuntimeError(f"Payload mismatch: {actual}, {drained}")
                samples.append(
                    {
                        "index": index,
                        "begin_mono_ns": begin,
                        "submit_return_mono_ns": submit_end,
                        "open_ms": (opened - begin) / 1e6,
                        "allocate_ms": (allocated - opened) / 1e6,
                        "submit_ms": (submit_end - allocated) / 1e6,
                        **drained,
                        "native_read_ms": (freed - begin) / 1e6,
                        "oracle_after_ms": (time.perf_counter_ns() - freed) / 1e6,
                        "checked_bytes": args.bytes,
                        "sha256": actual,
                    }
                )
            emit({"event": "read_complete", "id": message["id"], "samples": samples})
    finally:
        if registered and args.role == "source" and not source_revoked:
            wait_source_stop()
        if engine:
            for segment in segment_ids:
                checked(api.tent_close_segment(engine, segment), "close")
            if registered:
                checked(api.tent_unregister_memory(engine, address, args.bytes), "unregister")
            api.tent_destroy_engine(engine)
        del owner
        buffer.close()
        del libraries
        emit({"event": "stopped", "registered_regions": 0, "active_batches": 0})


if __name__ == "__main__":
    main()
