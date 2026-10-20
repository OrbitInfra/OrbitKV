"""Real installed TENT owner controls, kept out of the source-only test gate."""

import ctypes
import gc
import hashlib
import json
import os
import sys
import traceback
import weakref
from pathlib import Path


def rejected(operation, exception=Exception):
    try:
        operation()
    except exception as error:
        return str(error)
    raise AssertionError("operation unexpectedly succeeded")


def tcp_payload(native, extension, case):
    size, payload, stride = 32768, 1024, 4096
    source = (ctypes.c_ubyte * size)()
    destination = (ctypes.c_ubyte * size)()
    attempted = []
    control = None
    if case == "tcp-payload-register-error":
        os.environ["ORBITKV_TEST_SOURCE_OWNER_CASE"] = "partial"
        control = ctypes.CDLL(os.environ["ORBITKV_TENT_OWNER_CONTROL_LIBRARY"])
        for allocation in (source, destination):
            weakref.finalize(
                allocation,
                print,
                json.dumps({"case": case, "allocation_released_before_exit": True}),
                flush=True,
            )
    try:
        sender = native.MooncakeTransferEngine(bind_host="127.0.0.1")
        receiver = native.MooncakeTransferEngine(bind_host="127.0.0.1")
        for engine, allocation in [(sender, source), (receiver, destination)]:
            address = ctypes.addressof(allocation)
            attempted.append(address)
            engine.register_memory_borrowed([{"addr": address, "len": size, "location": "cpu:0"}])
        for round_number in range(3):
            for operation in ["write", "read"]:
                source[:] = [(index * 17 + round_number * 31) % 251 for index in range(size)]
                destination[:] = [0x5A] * size
                engine, endpoint = (
                    (sender, receiver.endpoint)
                    if operation == "write"
                    else (receiver, sender.endpoint)
                )
                local, remote = (
                    (source, destination) if operation == "write" else (destination, source)
                )
                slices = [
                    (ctypes.addressof(local) + offset, ctypes.addressof(remote) + offset, payload)
                    for offset in range(0, size, stride)
                ]
                original_source = bytes(source)
                expected = bytearray([0x5A] * size)
                for offset in range(0, size, stride):
                    expected[offset : offset + payload] = original_source[offset : offset + payload]
                completed = getattr(engine, operation)(endpoint, slices, timeout_s=5)
                assert bytes(source) == original_source
                assert completed == len(slices) * payload
                assert bytes(destination) == expected, (
                    operation,
                    round_number,
                    hashlib.sha256(bytes(destination)).hexdigest(),
                    hashlib.sha256(expected).hexdigest(),
                )
        maps = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(name in line for name in ["libtent_shared", "libmooncake_common", "libasio"])
        }
        assert len(maps) == 3, maps
        assert all("site-packages" in Path(path).parts for path in maps), maps
        sender.unregister_memory([attempted[0]])
        receiver.unregister_memory([attempted[1]])
        print(
            json.dumps(
                {
                    "case": "tcp-payload",
                    "passed": True,
                    "extension": str(extension),
                    "operations": 6,
                    "bytes_checked_per_operation": size,
                    "native_maps": sorted(maps),
                    "physical_GPU_or_RDMA_qualification": False,
                }
            ),
            flush=True,
        )
    except BaseException as error:
        # Unknown native work keeps this frame's allocations alive until process exit.
        try:
            traceback.print_exc()
            if control is not None:
                counters = {}
                for name in ("registers", "unregisters", "source_closes", "destroys"):
                    function = getattr(control, "orbitkv_test_" + name)
                    function.restype = ctypes.c_uint64
                    counters[name] = function()
                print(
                    json.dumps(
                        {
                            "case": case,
                            "passed": False,
                            "first_error": str(error),
                            "attempted_regions": len(attempted),
                            "native_counts": counters,
                            "physical_GPU_or_RDMA_qualification": False,
                        }
                    ),
                    flush=True,
                )
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(1)


def main():
    if sys.flags.optimize:
        raise RuntimeError("owner controls require nonoptimized Python")
    installation, case = Path(sys.argv[1]).resolve(strict=True), sys.argv[2]
    sys.path.insert(0, str(installation))
    import orbitkv.orbitkv as native

    extension = Path(native.__file__).resolve(strict=True)
    assert extension.parent == installation / "orbitkv" and "site-packages" in extension.parts
    assert "torch" not in sys.modules
    if case.startswith("tcp-payload"):
        tcp_payload(native, extension, case)
        return
    modes = {
        "normal": "normal",
        "invalid": "normal",
        "retry": "unregister-after-first",
        "gc-after-close-error": "unregister-error",
        "bulk-register-error": "partial",
        "native-partial": "native-register-partial",
        "borrowed-error": "unregister-error",
    }
    os.environ["ORBITKV_TEST_SOURCE_OWNER_CASE"] = modes[case]
    control_path = Path(os.environ["ORBITKV_TENT_OWNER_CONTROL_LIBRARY"]).resolve(strict=True)
    control = ctypes.CDLL(str(control_path))

    def count(name):
        function = getattr(control, "orbitkv_test_" + name)
        function.restype = ctypes.c_uint64
        return function()

    engine_type = native.MooncakeTransferEngine
    assert engine_type.memory_usage() == (0, 0, 0)
    engine = engine_type(bind_host="127.0.0.1")
    first = bytearray((index * 17 + 31) % 251 for index in range(8192))
    second = bytearray([0x5A] * 8192)
    if case == "borrowed-error":
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(first))
        engine.register_memory_borrowed([{"addr": address, "len": len(first), "location": "cpu:0"}])
        error = rejected(lambda: engine.unregister_memory([address]))
        del engine
        gc.collect()
        assert (count("registers"), count("unregisters"), count("source_closes")) == (1, 1, 0)
        assert count("destroys") == 0
        assert engine_type.memory_usage() == (1, len(first), 1)
        assert first[0] == 31
    elif case == "invalid":
        for views in [
            [memoryview(b"immutable")],
            [memoryview(first)[::2]],
            [memoryview(first).toreadonly()],
            [memoryview(first)[:16], memoryview(first)[8:24]],
        ]:
            rejected(lambda views=views: engine.register_memory(views))
            assert count("registers") == 0
            assert engine_type.memory_usage() == (0, 0, 0)
        rejected(lambda: engine.register_memory([memoryview(first)] * 1025))
        assert count("registers") == 0
        engine.close()
        del engine
        gc.collect()
        assert count("destroys") == 1
        error = None
    else:
        view = memoryview(first)[1024:5120]
        other = memoryview(second)[2048:6144]
        alive = weakref.ref(view)
        other_alive = weakref.ref(other)
        capacity = first.__alloc__() + second.__alloc__()
        if case == "native-partial":
            error = rejected(lambda: engine.register_memory([view]))
            del engine, view, other
            gc.collect()
            assert alive() is not None and other_alive() is None
            rejected(lambda: first.extend(b"x"), BufferError)
            pending = control.orbitkv_test_pending_registration
            pending.restype = ctypes.c_int
            assert pending() == 1, "real native partially retained provider was not established"
            assert engine_type.memory_usage() == (1, first.__alloc__(), 1)
            assert (count("registers"), count("unregisters"), count("source_closes")) == (1, 0, 0)
            assert count("destroys") == 0
        elif case == "bulk-register-error":
            error = rejected(lambda: engine.register_memory([view, other]))
            del engine, view, other
            gc.collect()
            assert alive() is not None and other_alive() is not None
            rejected(lambda: first.extend(b"x"), BufferError)
            rejected(lambda: second.extend(b"x"), BufferError)
            assert engine_type.memory_usage() == (2, capacity, 1)
            assert (count("registers"), count("unregisters"), count("source_closes")) == (2, 0, 0)
            assert count("destroys") == 0
        else:
            addresses = engine.register_memory([view, other])
            assert engine_type.memory_usage() == (2, capacity, 0)
            assert capacity > len(view) + len(other)
            rejected(view.release, BufferError)
            rejected(other.release, BufferError)
            del view, other
            gc.collect()
            assert alive() is not None and other_alive() is not None
            rejected(lambda: first.extend(b"x"), BufferError)
            if case == "normal":
                before = bytes(first)
                assert engine.write(engine.endpoint, [(addresses[0], addresses[1], 4096)]) == 4096
                assert bytes(first) == before
                assert second[:2048] == bytes([0x5A]) * 2048
                assert second[2048:6144] == first[1024:5120]
                assert second[6144:] == bytes([0x5A]) * 2048
                rejected(
                    lambda: engine.read(engine.endpoint, [(addresses[0] - 1, addresses[1], 1)])
                )
                engine.close()
                engine.close()
                assert count("unregisters") == 2
                assert engine_type.memory_usage() == (0, 0, 0)
                assert alive() is None and other_alive() is None
                first.extend(b"x")
                rejected(lambda: engine.register_memory([memoryview(first)]))
                del engine
                gc.collect()
                assert count("destroys") == 1
                fresh = engine_type(bind_host="127.0.0.1")
                fresh.register_memory([memoryview(first)])
                del fresh
                gc.collect()
                assert engine_type.memory_usage() == (0, 0, 0)
                first.extend(b"x")
                assert count("destroys") == 2
                error = None
            else:
                error = rejected(engine.close)
                assert engine_type.memory_usage() == (2, capacity, 0)
                assert alive() is not None and other_alive() is not None
                rejected(lambda: engine.register_memory([memoryview(bytearray(16))]))
                if case == "retry":
                    assert count("unregisters") == 2
                    os.environ["ORBITKV_TEST_SOURCE_OWNER_CASE"] = "normal"
                    engine.close()
                    assert count("unregisters") == 3, "successful prefix cleanup was repeated"
                    assert engine_type.memory_usage() == (0, 0, 0)
                    assert alive() is None and other_alive() is None
                    del engine
                    gc.collect()
                    assert count("destroys") == 1
                else:
                    assert case == "gc-after-close-error"
                    del engine
                    gc.collect()
                    assert count("source_closes") == 1, "GC retried a failed explicit close"
                    assert count("unregisters") == 1
                    assert count("destroys") == 0
                    assert engine_type.memory_usage() == (2, capacity, 1)
                    assert alive() is not None and other_alive() is not None
    maps = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(name in line for name in ("libtent", "libmooncake_common", "libasio"))
        }
    )
    assert str(control_path) in maps
    print(
        json.dumps(
            {
                "case": case,
                "passed": True,
                "extension": str(extension),
                "usage": engine_type.memory_usage(),
                "first_error": error,
                "registers": count("registers"),
                "unregisters": count("unregisters"),
                "source_closes": count("source_closes"),
                "destroys": count("destroys"),
                "native_maps": maps,
                "physical_GPU_or_RDMA_qualification": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
