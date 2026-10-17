"""Manager test launches keep source and installed package modes distinct."""

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tests.support import cache_manager, registration_campaign
from tests.support.cache_manager import CacheManagerProcess, manager_pythonpath
from tests.support.paths import PYTHON_ROOT
from tests.support.vllm_helpers import CacheManager, VLLMServer


def test_manager_pythonpath_defaults_to_source_then_inherited_paths():
    paths = manager_pythonpath(
        ["/caller/python", "/installed/site"],
        ["/installed/site", "/system/site"],
        None,
    )
    assert paths == (
        str(PYTHON_ROOT),
        "/caller/python",
        "/installed/site",
        "/system/site",
    )


def test_manager_pythonpath_installed_mode_excludes_source_and_inherited_paths():
    paths = manager_pythonpath(
        [str(PYTHON_ROOT), "/unrelated/source"],
        ["/installed/site", "/system/site"],
        ("/installed/site",),
    )
    assert paths == ("/installed/site", "/system/site")
    assert str(PYTHON_ROOT) not in paths


def test_vllm_manager_keeps_gpu_mask_and_installed_package_priority(monkeypatch):
    launched = {}

    def start(command, **kwargs):
        launched.update(kwargs)
        return SimpleNamespace(pid=1)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-target")
    monkeypatch.setenv("PYTHONPATH", "/installed-wheel:/tests")
    monkeypatch.setattr("tests.support.vllm_helpers.subprocess.Popen", start)
    monkeypatch.setattr(CacheManager, "_wait_for_ready", lambda _: None)
    manager = CacheManager(server_binary="/installed-wheel/manager", cargo_features=[])
    manager.__enter__()
    assert launched["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-target"
    assert launched["env"]["PYTHONPATH"].split(":")[:2] == ["/installed-wheel", "/tests"]


def test_vllm_worker_keeps_installed_package_priority(monkeypatch):
    launched = {}

    def start(command, **kwargs):
        launched.update(kwargs)
        return SimpleNamespace(pid=1)

    monkeypatch.setenv("PYTHONPATH", "/installed-wheel:/tests")
    monkeypatch.setattr("tests.support.vllm_helpers.subprocess.Popen", start)
    monkeypatch.setattr(VLLMServer, "_wait_for_ready", lambda *args, **kwargs: None)
    VLLMServer(model="/test/model", port=12345).__enter__()
    assert launched["env"]["PYTHONPATH"].split(":")[:2] == ["/installed-wheel", "/tests"]


@pytest.mark.parametrize("write_failure", [False, True])
def test_strict_manager_start_records_actual_command_and_keeps_failed_record_owner(
    tmp_path, monkeypatch, write_failure
):
    actual = tmp_path / "actual.json"
    binary = tmp_path / "cpu-manager"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json,os,signal,sys\nfrom pathlib import Path\n"
        "def stop(*_):\n print('SIGTERM received, shutting down', flush=True)\n sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"Path({str(actual)!r}).write_text(json.dumps({{'argv':sys.argv,'pid':os.getpid()}}))\n"
        "while True: signal.pause()\n"
    )
    binary.chmod(0o755)
    monkeypatch.setattr(cache_manager, "find_cache_manager_binary", lambda: str(binary))
    monkeypatch.setattr(cache_manager, "wait_for_server_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setenv("ORBITKV_TEST_STRICT_SHUTDOWN", "1")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("MC_FORCE_TCP", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-CPU-fixture")
    server = CacheManagerProcess(
        port=12345,
        http_port=12346,
        channel_service="CPU-fixture",
        channel_session_epoch=17,
        bootstrap_socket=str(tmp_path / "fixture.sock"),
        log_path=tmp_path / "manager.log",
    )
    write = registration_campaign.write

    def persist(path, value, deadline=None):
        if write_failure and path.name.endswith("manager-start.json"):
            raise OSError("controlled Manager startup record failure")
        write(path, value, deadline)

    monkeypatch.setattr(registration_campaign, "write", persist)
    try:
        if write_failure:
            with pytest.raises(OSError, match="startup record failure"):
                server.start()
            assert server.process is not None and server.process.poll() is None
        else:
            assert server.start()
        deadline = time.monotonic() + 5
        while not actual.exists():
            assert server.process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        observed = json.loads(actual.read_text())
        assert observed["pid"] == server.process.pid
        if not write_failure:
            executable = cache_manager.Path(f"/proc/{server.process.pid}/exe").resolve()
            raw = cache_manager.Path(f"/proc/{server.process.pid}/maps").read_text()
            unavailable_identity = False
            for line in raw.splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    path = cache_manager.Path(fields[5])
                    if ".so" in path.name or path == executable:
                        major, minor = (int(part, 16) for part in fields[3].split(":"))
                        metadata = path.stat()
                        unavailable_identity |= (os.makedev(major, minor), int(fields[4])) != (
                            metadata.st_dev,
                            metadata.st_ino,
                        )
            if unavailable_identity:
                server.record_runtime()
                runtime = json.loads(next(tmp_path.glob("*.manager-runtime.json")).read_text())
                for row in runtime["files"]:
                    if row["device"] != row["mapped_device"]:
                        assert row["executable_regions"]
                        assert all(
                            region["file_sha256"] == region["mapped_sha256"]
                            for region in row["executable_regions"]
                        )
            else:
                server.record_runtime()
                runtime = json.loads(next(tmp_path.glob("*.manager-runtime.json")).read_text())
                assert runtime["owner"]["pid"] == server.process.pid
                assert runtime["owner"]["start_ticks"] == int(server._process_start_ticks)
                assert runtime["maps"] == raw
                assert runtime["physical_termination_claimed"] is False
                assert runtime["files"] and all(
                    len(row["sha256"]) == 64 and row["bytes"] > 0 for row in runtime["files"]
                )
            records = list(tmp_path.glob("*.manager-start.json"))
            assert len(records) == 1
            record = json.loads(records[0].read_text())
            assert record["argv"] == observed["argv"] == list(server.command)
            assert record["pid"] == observed["pid"]
            assert record["start_ticks"] == int(server._process_start_ticks)
            assert record["bootstrap_socket"] == str(tmp_path / "fixture.sock")
            assert record["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-CPU-fixture"
    finally:
        process = server.process
        server.stop()
        cleanup = json.loads((tmp_path / "manager.log.cleanup.json").read_text())
        assert cleanup["pid"] == process.pid and cleanup["sigterm_sent"] is True
        assert cleanup["exit_code"] == 0 and cleanup["reaped"] is True
        assert cleanup["forced_kill"] is False
        assert "SIGTERM received, shutting down" in (tmp_path / "manager.log").read_text()


@pytest.mark.parametrize("replace_at_open", [False, True])
def test_runtime_hash_is_bound_to_the_opened_mapped_inode(tmp_path, monkeypatch, replace_at_open):
    library = tmp_path / "mapped.so"
    library.write_bytes(b"A" * 4096)
    replacement = tmp_path / "replacement.so"
    replacement.write_bytes(b"B" * 4096)
    backup = tmp_path / "original.so"
    ready = tmp_path / "ready"
    argv = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        "import mmap,signal,sys\nfrom pathlib import Path\n"
        "f=open(sys.argv[1],'rb');m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)\n"
        "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))\n"
        "Path(sys.argv[2]).write_text('ready')\nwhile True:signal.pause()\n",
        str(library),
        str(ready),
    ]
    server = CacheManagerProcess(port=12345, log_path=tmp_path / "manager.log")
    process = subprocess.Popen(argv, start_new_session=True)
    server.process = process
    server._process_start_ticks = str(registration_campaign.identity(process.pid)["start_ticks"])
    server._log_path = tmp_path / "manager.log"
    deadline = time.monotonic() + 5
    try:
        while not ready.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        paths = {}
        for line in cache_manager.Path(f"/proc/{process.pid}/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                major, minor = (int(part, 16) for part in fields[3].split(":"))
                paths[int(fields[4])] = os.makedev(major, minor)
        actual_fstat, actual_stat, actual_open = os.fstat, cache_manager.Path.stat, os.open

        def join(metadata):
            # CPU seam for overlay device numbers only; inode and file bytes are real.
            if metadata.st_ino not in paths:
                return metadata
            values = list(metadata)
            values[2] = paths[metadata.st_ino]
            return os.stat_result(values)

        def opened(path, flags, *args, **kwargs):
            if replace_at_open and cache_manager.Path(path) == library:
                library.rename(backup)
                replacement.rename(library)
                try:
                    return actual_open(path, flags, *args, **kwargs)
                finally:
                    library.rename(replacement)
                    backup.rename(library)
            return actual_open(path, flags, *args, **kwargs)

        with monkeypatch.context() as local:
            local.setattr(os, "fstat", lambda fd: join(actual_fstat(fd)))
            local.setattr(
                cache_manager.Path, "stat", lambda path, *a, **kw: join(actual_stat(path, *a, **kw))
            )
            local.setattr(os, "open", opened)
            if replace_at_open:
                with pytest.raises(RuntimeError, match="mapped runtime path was replaced"):
                    server.record_runtime()
                assert not list(tmp_path.glob("*.manager-runtime.json"))
            else:
                server.record_runtime()
                runtime = json.loads(next(tmp_path.glob("*.manager-runtime.json")).read_text())
                row = next(row for row in runtime["files"] if row["path"] == str(library))
                assert row["inode"] == library.stat().st_ino
                assert row["sha256"] == registration_campaign.digest(library)
            assert process.poll() is None
    finally:
        process.terminate()
        assert process.wait(timeout=5) == 0


@pytest.mark.parametrize("corrupt_mapping", [False, True, "short-read"])
def test_overlay_runtime_requires_actual_complete_executable_bytes(
    tmp_path, monkeypatch, corrupt_mapping
):
    library = tmp_path / "mapped-executable.so"
    library.write_bytes(b"A" * 4096)
    ready = tmp_path / "ready"
    code = (
        "import mmap,os,signal,sys\nfrom pathlib import Path\n"
        "f=open(sys.argv[1],'rb');m=mmap.mmap(f.fileno(),0,flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ|mmap.PROT_EXEC)\n"
        "if sys.argv[3] in ('1','2'):\n"
        " import ctypes\n"
        " # Test-only code page is never executed. Mutate the actual private mapping.\n"
        " libc=ctypes.CDLL(None);libc.mprotect.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]\n"
        " line=next(s for s in Path('/proc/self/maps').read_text().splitlines() if s.endswith(' '+sys.argv[1]))\n"
        " address=int(line.split('-')[0],16)\n"
        " if libc.mprotect(address,4096,mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)!=0:raise RuntimeError('mprotect failed')\n"
        " if sys.argv[3]=='1':ctypes.memset(address,66,4096)\n"
        " else:ctypes.memset(address+2048,0,2048)\n"
        " if libc.mprotect(address,4096,mmap.PROT_READ|mmap.PROT_EXEC)!=0:raise RuntimeError('mprotect restore failed')\n"
        "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))\n"
        "Path(sys.argv[2]).write_text('ready')\nwhile True:signal.pause()\n"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            code,
            str(library),
            str(ready),
            "2" if corrupt_mapping == "short-read" else str(int(corrupt_mapping)),
        ],
        start_new_session=True,
    )
    server = CacheManagerProcess(port=12345, log_path=tmp_path / "manager.log")
    server.process = process
    server._process_start_ticks = str(registration_campaign.identity(process.pid)["start_ticks"])
    server._log_path = tmp_path / "manager.log"
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        metadata = library.stat()
        line = next(
            s
            for s in cache_manager.Path(f"/proc/{process.pid}/maps").read_text().splitlines()
            if s.endswith(" " + str(library))
        )
        major, minor = (int(part, 16) for part in line.split()[3].split(":"))
        if metadata.st_dev == os.makedev(major, minor):
            pytest.skip("Control specifically tests differing overlay devices")
        if corrupt_mapping == "short-read":
            pread = os.pread

            def short(fd, count, offset):
                value = pread(fd, count, offset)
                if os.fstat(fd).st_ino == metadata.st_ino and count == 4096 and offset == 0:
                    return value[:2048]
                return value

            monkeypatch.setattr(os, "pread", short)
            with pytest.raises(RuntimeError, match="file read was short"):
                server.record_runtime()
            assert not list(tmp_path.glob("*.manager-runtime.json"))
        elif corrupt_mapping:
            with pytest.raises(RuntimeError, match="Actual mapped executable bytes differ"):
                server.record_runtime()
            assert not list(tmp_path.glob("*.manager-runtime.json"))
        else:
            server.record_runtime()
            runtime = json.loads(next(tmp_path.glob("*.manager-runtime.json")).read_text())
            row = next(row for row in runtime["files"] if row["path"] == str(library))
            assert row["device"] != row["mapped_device"]
            assert len(row["executable_regions"]) == 1
            region = row["executable_regions"][0]
            assert region["end"] - region["start"] == 4096
            assert region["file_sha256"] == region["mapped_sha256"]
        assert process.poll() is None
    finally:
        process.terminate()
        assert process.wait(timeout=5) == 0


@pytest.mark.parametrize("control", ["complete", "changed-tail", "over-bound"])
def test_large_runtime_code_proof_is_complete_and_bounded(tmp_path, monkeypatch, control):
    library = tmp_path / "large-executable.so"
    size = (
        registration_campaign.MAX_MAPPED_EXECUTABLE_BYTES + 4096
        if control == "over-bound"
        else 513 * 1024 * 1024
    )
    with library.open("wb") as file:
        file.truncate(size)
        file.seek(0)
        file.write(b"A" * 4096)
        file.seek(size - 4096)
        file.write(b"Z" * 4096)
    ready = tmp_path / "ready"
    code = (
        "import mmap,signal,sys\nfrom pathlib import Path\n"
        "f=open(sys.argv[1],'rb');m=mmap.mmap(f.fileno(),0,flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ|mmap.PROT_EXEC)\n"
        "if sys.argv[3]=='changed-tail':\n"
        " import ctypes\n"
        " libc=ctypes.CDLL(None);libc.mprotect.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]\n"
        " line=next(s for s in Path('/proc/self/maps').read_text().splitlines() if s.endswith(' '+sys.argv[1]))\n"
        " address=int(line.split('-')[0],16)+len(m)-4096\n"
        " if libc.mprotect(address,4096,mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC)!=0:raise RuntimeError('mprotect failed')\n"
        " ctypes.memset(address,66,4096)\n"
        " if libc.mprotect(address,4096,mmap.PROT_READ|mmap.PROT_EXEC)!=0:raise RuntimeError('mprotect restore failed')\n"
        "signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))\n"
        "Path(sys.argv[2]).write_text('ready')\nwhile True:signal.pause()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(library), str(ready), control],
        start_new_session=True,
    )
    server = CacheManagerProcess(port=12345, log_path=tmp_path / "manager.log")
    server.process = process
    server._process_start_ticks = str(registration_campaign.identity(process.pid)["start_ticks"])
    server._log_path = tmp_path / "manager.log"
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        line = next(
            s
            for s in cache_manager.Path(f"/proc/{process.pid}/maps").read_text().splitlines()
            if s.endswith(" " + str(library))
        )
        major, minor = (int(part, 16) for part in line.split()[3].split(":"))
        if library.stat().st_dev == os.makedev(major, minor):
            pytest.skip("Control requires an actual differing overlay device view")
        pread = os.pread
        reads = []

        def observed_read(fd, count, offset):
            assert count <= 1024 * 1024
            reads.append(count)
            return pread(fd, count, offset)

        monkeypatch.setattr(os, "pread", observed_read)
        if control == "over-bound":
            with pytest.raises(RuntimeError, match="exceeds byte bound"):
                server.record_runtime()
            assert reads == []
        elif control == "changed-tail":
            with pytest.raises(RuntimeError, match="Actual mapped executable bytes differ"):
                server.record_runtime()
            assert sum(reads) >= 2 * size
        else:
            server.record_runtime()
            runtime = json.loads(next(tmp_path.glob("*.manager-runtime.json")).read_text())
            row = next(row for row in runtime["files"] if row["path"] == str(library))
            assert sum(part["end"] - part["start"] for part in row["executable_regions"]) == size
            assert all(
                part["file_sha256"] == part["mapped_sha256"] for part in row["executable_regions"]
            )
            assert sum(reads) >= 2 * size
        if control != "complete":
            assert not list(tmp_path.glob("*.manager-runtime.json"))
        assert process.poll() is None
    finally:
        process.terminate()
        assert process.wait(timeout=5) == 0
