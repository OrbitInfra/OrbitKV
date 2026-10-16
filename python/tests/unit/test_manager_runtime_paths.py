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
                with pytest.raises(RuntimeError, match="mapped runtime path was replaced"):
                    server.record_runtime()
                assert not list(tmp_path.glob("*.manager-runtime.json"))
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
