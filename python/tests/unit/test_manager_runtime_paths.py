"""Manager test launches keep source and installed package modes distinct."""

from types import SimpleNamespace

from tests.support.cache_manager import manager_pythonpath
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
