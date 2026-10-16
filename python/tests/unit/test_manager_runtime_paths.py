"""Manager test launches keep source and installed package modes distinct."""

import json
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
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
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
            records = list(tmp_path.glob("*.manager-start.json"))
            assert len(records) == 1
            record = json.loads(records[0].read_text())
            assert record["argv"] == observed["argv"] == list(server.command)
            assert record["pid"] == observed["pid"]
            assert record["start_ticks"] == int(server._process_start_ticks)
            assert record["bootstrap_socket"] == str(tmp_path / "fixture.sock")
            assert record["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-CPU-fixture"
    finally:
        result = server.stop_without_kill(timeout=5)
        assert result["status"] == "exited" and result["returncode"] == 0
        assert result["reaped"] is True and result["forced_cleanup"] is False
