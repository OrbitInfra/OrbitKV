"""Observe OrbitKV export credits during an installed SGLang normal shutdown."""

import struct
from pathlib import Path

# PyTorch RefcountedMapAllocator places MapInfo before the int64 counter array.
COUNTER_DATA_OFFSET = 64


def read_counter(path: Path, offset: int) -> int:
    if offset < 0:
        raise ValueError("CUDA IPC counter offset must be nonnegative")
    with path.open("rb") as stream:
        stream.seek(COUNTER_DATA_OFFSET + offset * 8)
        data = stream.read(8)
    if len(data) != 8:
        raise ValueError("CUDA IPC counter slot is missing or truncated")
    return struct.unpack("=q", data)[0]


def install_trace() -> None:
    import atexit
    import json
    import os
    import time

    import orbitkv.sglang.linker as linker_module

    trace = Path(os.environ["ORBITKV_CLOSE_TRACE_DIRECTORY"])
    exports = []
    original_close = linker_module.OrbitKVLinker.close

    def emit(stage, **details):
        event = {
            "stage": stage,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
            "counter_data_offset_bytes": COUNTER_DATA_OFFSET,
            **details,
        }
        with (trace / f"{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def counters():
        readings = []
        for name, offset in exports:
            path = Path("/dev/shm") / name.lstrip("/")
            try:
                readings.append(
                    {"file": name, "offset": offset, "value": read_counter(path, offset)}
                )
            except (OSError, ValueError) as error:
                readings.append({"file": name, "offset": offset, "error": str(error)})
        return readings

    def close(self):
        for name, offset in self.registration.ipc_counters:
            value = name.decode() if isinstance(name, bytes) else name
            if (value, offset) not in exports:
                exports.append((value, offset))
        emit("linker_close_start", instance_id=self.instance_id, counters=counters())
        try:
            result = original_close(self)
        except BaseException as error:
            emit("linker_close_error", error=repr(error), counters=counters())
            raise
        emit("linker_close_return", instance_id=self.instance_id, counters=counters())
        return result

    linker_module.OrbitKVLinker.close = close
    atexit.register(lambda: emit("python_atexit", exports=len(exports), counters=counters()))


if __name__ in {"__main__", "__mp_main__"}:
    install_trace()

if __name__ == "__main__":
    import runpy

    runpy.run_module("sglang.launch_server", run_name="__main__")
