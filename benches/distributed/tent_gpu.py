"""Owned GPU-pointer READ/WRITE endpoint for a frozen two-host TENT experiment."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import json
import os
import signal
import stat
import sys
import time
import traceback
import uuid
from pathlib import Path

CAPACITY = 64 * 1024 * 1024


def slices(local: int, remote: int, payload: int, chunk: int, stride: int) -> list[tuple]:
    values = (local, remote, payload, chunk, stride)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Addresses and geometry must be positive integers")
    if (
        payload > CAPACITY
        or payload % chunk
        or chunk % 4
        or stride % 4
        or stride < chunk
        or (payload // chunk - 1) * stride + chunk > CAPACITY
        or payload // chunk > 8192
        or max(local, remote) + CAPACITY >= 2**64
    ):
        raise ValueError("Slices exceed the registered GPU buffer")
    return [
        (local + index * stride, remote + index * stride, chunk)
        for index in range(payload // chunk)
    ]


def validate_stop(message: dict, role: str, cell: str, peer_host: str) -> None:
    if role != "peer":
        return
    receipt = message.get("peer_receipt", {})
    if not (
        receipt.get("cell") == cell
        and receipt.get("host") == peer_host
        and receipt.get("name") == "gpu-reader"
        and receipt.get("native_stopped") is True
        and type(receipt.get("exit_code")) is int
        and receipt["exit_code"] in (0, 1)
        and receipt.get("actual_wait_reap") is True
        and receipt.get("forced_cleanup") is False
    ):
        raise ValueError("Peer stop requires the same-cell reader's actual drain/reap receipt")


def emit(value: dict) -> None:
    print(json.dumps({"tent_gpu_endpoint": True, **value}), flush=True)


def retain(error: str) -> None:
    with contextlib.suppress(OSError):
        emit({"event": "ownership_unknown", "error": error, "ownership_held": True})
    while True:
        signal.pause()


class GpuBuffers:
    """Keep device allocations and oracle scratch until explicit native drain."""

    def __init__(self, gpu: int, oracle: Path):
        self.cuda = ctypes.CDLL("libcudart.so.13")
        self.oracle = ctypes.CDLL(str(oracle))
        self.buffers: list[ctypes.c_void_p] = []
        self.gpu = gpu
        signatures = {
            "cudaGetErrorString": ([ctypes.c_int], ctypes.c_char_p),
            "cudaSetDevice": ([ctypes.c_int], ctypes.c_int),
            "cudaMalloc": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
            "cudaFree": ([ctypes.c_void_p], ctypes.c_int),
            "cudaMemset": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t], ctypes.c_int),
            "cudaDeviceSynchronize": ([], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.cuda, name)
            function.argtypes, function.restype = arguments, result
        self.oracle.tent_gpu_fill.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
        self.oracle.tent_gpu_corrupt.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.oracle.tent_gpu_pointer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.oracle.tent_gpu_uuid.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.oracle.tent_gpu_verify.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]

    def check(self, result: int, operation: str) -> None:
        if result:
            raise RuntimeError(f"{operation}: {self.cuda.cudaGetErrorString(result).decode()}")

    def allocate(self) -> list[dict]:
        self.check(self.cuda.cudaSetDevice(self.gpu), "select GPU")
        raw_uuid = (ctypes.c_ubyte * 16)()
        self.check(self.oracle.tent_gpu_uuid(self.gpu, raw_uuid), "CUDA GPU UUID")
        gpu_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid)))
        attributes = []
        for size in [CAPACITY, 16]:
            pointer = ctypes.c_void_p()
            self.check(self.cuda.cudaMalloc(ctypes.byref(pointer), size), "allocate GPU")
            self.buffers.append(pointer)
            device, kind = ctypes.c_int(), ctypes.c_int()
            self.check(
                self.oracle.tent_gpu_pointer(pointer, ctypes.byref(device), ctypes.byref(kind)),
                "GPU pointer attributes",
            )
            if device.value != self.gpu or kind.value != 2:
                raise RuntimeError("Address is not device memory on the declared GPU")
            attributes.append(
                {
                    "address": pointer.value,
                    "bytes": size,
                    "gpu": device.value,
                    "uuid": gpu_uuid,
                    "memory_type": kind.value,
                }
            )
        return attributes

    def fill(self, buffer: int, seed: int) -> None:
        self.check(self.cuda.cudaSetDevice(self.gpu), "select fill GPU")
        self.check(self.oracle.tent_gpu_fill(self.buffers[buffer], CAPACITY, seed), "GPU fill")

    def clear(self, buffer: int) -> None:
        self.check(self.cuda.cudaSetDevice(self.gpu), "select clear GPU")
        self.check(self.cuda.cudaMemset(self.buffers[buffer], 165, CAPACITY), "GPU clear")
        self.check(self.cuda.cudaDeviceSynchronize(), "GPU clear drain")

    def verify(self, buffer: int, seed: int, payload: int, chunk: int, stride: int) -> dict:
        self.check(self.cuda.cudaSetDevice(self.gpu), "select oracle GPU")
        cpu, gpu = (ctypes.c_uint64 * 2)(), (ctypes.c_uint64 * 2)()
        self.check(
            self.oracle.tent_gpu_verify(
                self.buffers[buffer],
                CAPACITY,
                seed,
                payload,
                chunk,
                stride,
                self.buffers[1],
                cpu,
                gpu,
            ),
            "complete CPU/GPU byte oracle",
        )
        return {
            "cpu_mismatches": cpu[0],
            "gpu_mismatches": gpu[0],
            "cpu_first_word": cpu[1],
            "gpu_first_word": gpu[1],
            "checked_bytes": CAPACITY,
            "payload_bytes": payload,
            "sentinel_bytes": CAPACITY - payload,
        }

    def close(self) -> None:
        self.check(self.cuda.cudaSetDevice(self.gpu), "cleanup GPU")
        while self.buffers:
            self.check(self.cuda.cudaFree(self.buffers[-1]), "free drained GPU allocation")
            self.buffers.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-directory", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--nics", required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--role", choices=("reader", "peer"), required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--peer-host", required=True)
    parser.add_argument("--resource-lock-fd", type=int, action="append", required=True)
    args = parser.parse_args()
    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        raise RuntimeError("Optimized guards forbidden")
    if len(set(args.resource_lock_fd)) != len(args.resource_lock_fd) or any(
        fd < 3 or not stat.S_ISREG(os.fstat(fd).st_mode) for fd in args.resource_lock_fd
    ):
        raise RuntimeError("Valid inherited resource leases are required")
    sys.path.insert(0, str(args.installed_directory))
    import orbitkv.orbitkv as native

    if not Path(native.__file__).resolve().is_relative_to(args.installed_directory.resolve()):
        raise RuntimeError("Native extension is outside the frozen installed artifact")
    memory = GpuBuffers(args.gpu, args.oracle)
    engine = None
    registrations = []
    exposed = stopped = failed = False
    seen = set()
    try:
        attributes = memory.allocate()
        begin = time.monotonic_ns()
        engine = native.MooncakeTransferEngine(bind_host=args.bind, nics=args.nics.split(","))
        initialized = time.monotonic_ns()
        for pointer in memory.buffers[:1]:
            registrations.append(pointer.value)
            exposed = True
            try:
                engine.register_memory_borrowed(
                    [{"addr": pointer.value, "len": CAPACITY, "location": "*"}]
                )
            except BaseException as error:
                traceback.print_exc(file=sys.stderr)
                retain(f"Registration attempt has unproven native ownership: {error}")
        registered = time.monotonic_ns()
        exposed = True
        emit(
            {
                "event": "ready",
                "pid": os.getpid(),
                "endpoint": engine.endpoint,
                "addresses": registrations,
                "capacity": CAPACITY,
                "attributes": attributes,
                "initialize_ms": (initialized - begin) / 1e6,
                "register_ms": (registered - initialized) / 1e6,
                "installed_extension": native.__file__,
                "native_maps": sorted(
                    {
                        line.split()[-1]
                        for line in Path("/proc/self/maps").read_text().splitlines()
                        if ".so" in line and "/" in line
                    }
                ),
            }
        )
        while True:
            line = sys.stdin.readline(4097)
            if not line:
                retain("Control EOF cannot revoke exported GPU memory")
            if len(line) > 4096:
                retain("Oversized control frame")
            message = {}
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("Control frame must be an object")
                request_id = message["id"]
                if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                    raise ValueError("Invalid request id")
                if request_id in seen or len(seen) >= 1024:
                    raise ValueError("Duplicate or exhausted bounded command ids")
                seen.add(request_id)
                operation = message["operation"]
                if failed and operation != "stop":
                    raise RuntimeError("Failed endpoint accepts only explicit drain/stop")
                if operation == "stop":
                    validate_stop(message, args.role, args.cell, args.peer_host)
                    stopped = True
                    break
                buffer = message.get("buffer", 0)
                seed = message.get("seed", 1)
                if type(buffer) is not int or buffer != 0:
                    raise ValueError("Invalid registered buffer")
                if type(seed) is not int or not 0 <= seed < 2**32:
                    raise ValueError("Invalid deterministic seed")
                payload = message.get("payload", 32 * 1024 * 1024)
                chunk = message.get("chunk", payload)
                stride = message.get("stride", chunk)
                slices(
                    memory.buffers[buffer].value,
                    memory.buffers[buffer].value,
                    payload,
                    chunk,
                    stride,
                )
                if operation == "fill":
                    memory.fill(buffer, seed)
                    result = {"filled_bytes": CAPACITY}
                elif operation == "clear":
                    memory.clear(buffer)
                    result = {"cleared_bytes": CAPACITY}
                elif operation in ("verify", "oracle-control"):
                    result = memory.verify(buffer, seed, payload, chunk, stride)
                    if result["cpu_mismatches"] or result["gpu_mismatches"]:
                        raise AssertionError(result)
                    if operation == "oracle-control":
                        last = ((payload // chunk - 1) * stride + chunk) // 4 - 1
                        memory.check(
                            memory.oracle.tent_gpu_corrupt(memory.buffers[buffer], last),
                            "intentional GPU word corruption",
                        )
                        rejected = memory.verify(buffer, seed, payload, chunk, stride)
                        if rejected["cpu_mismatches"] != 1 or rejected["gpu_mismatches"] != 1:
                            raise AssertionError(rejected)
                        result = {"negative_control": True, "before": result, "after": rejected}
                elif operation in ("read", "write") and args.role == "reader":
                    transfers = slices(
                        memory.buffers[buffer].value,
                        message["remote_address"],
                        payload,
                        chunk,
                        stride,
                    )
                    begin = time.monotonic_ns()
                    completed = getattr(engine, operation)(
                        message["endpoint"], transfers, timeout_s=30
                    )
                    terminal = time.monotonic_ns()
                    if completed != payload:
                        raise AssertionError(f"Completed {completed}, expected {payload}")
                    stats = engine.nic_load_stats()
                    if any(inflight for _, inflight, _ in stats):
                        raise AssertionError("Native NIC work did not drain")
                    result = {
                        "completed_bytes": completed,
                        "descriptors": len(transfers),
                        "binding_to_terminal_ms": (terminal - begin) / 1e6,
                        "begin_mono_ns": begin,
                        "terminal_mono_ns": terminal,
                        "nic_load_stats": stats,
                    }
                else:
                    raise ValueError("Unknown or role-ineligible operation")
                emit({"event": "result", "id": request_id, "ok": True, **result})
            except Exception as error:
                failed = True
                emit(
                    {
                        "event": "result",
                        "id": message.get("id") if isinstance(message, dict) else None,
                        "ok": False,
                        "error": str(error),
                        "error_type": type(error).__name__,
                    }
                )
                traceback.print_exc(file=sys.stderr)
    finally:
        if exposed and not stopped:
            retain("Unexpected exit cannot revoke an exported region")
        try:
            if engine:
                if any(inflight for _, inflight, _ in engine.nic_load_stats()):
                    retain("Native work is still outstanding")
                if registrations:
                    engine.unregister_memory(registrations)
                    registrations.clear()
                engine = None
                gc.collect()
            memory.close()
        except BaseException as error:
            retain(f"Cannot establish GPU unregister/free: {error}")
        emit(
            {
                "event": "stopped",
                "native_stopped": True,
                "registered_regions": 0,
                "gpu_allocations": 0,
                "failed": failed,
            }
        )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
