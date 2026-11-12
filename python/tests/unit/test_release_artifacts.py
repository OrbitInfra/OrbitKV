"""Release gates use the complete wheel matrix and exact registry receipts."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from urllib.error import URLError
from zipfile import ZipFile, ZipInfo

import pytest


@pytest.fixture
def artifacts(tmp_path):
    path = Path(__file__).resolve().parents[3] / "scripts/check-wheel.py"
    spec = importlib.util.spec_from_file_location("release_artifacts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = [
        "__init__.py",
        "orbitkv.pyi",
        "client/gpu.py",
        "orbitkv.cpython.so",
        "vllm/plugin.py",
        "vllm/connector.py",
        "vllm/config.py",
        "vllm/layout.py",
        "vllm/metadata.py",
        "vllm/metrics.py",
        "sglang/plugin.py",
        "sglang/linker.py",
        "sglang/config.py",
        "sglang/layout.py",
    ]
    info = "orbitkv-0.1.0b1.dist-info"
    for minor in range(10, 15):
        for arch in ("x86_64", "aarch64"):
            name = f"orbitkv-0.1.0b1-cp3{minor}-cp3{minor}-manylinux_2_28_{arch}.whl"
            with ZipFile(tmp_path / name, "w") as wheel:
                for member in paths:
                    wheel.writestr("orbitkv/" + member, b"fixture")
                manager = ZipInfo("orbitkv/orbitkv-cache-manager-py")
                manager.external_attr = 0o100755 << 16
                wheel.writestr(manager, b"fixture, not an executable")
                wheel.writestr(
                    info + "/METADATA",
                    "Name: orbitkv\nVersion: 0.1.0b1\n"
                    "Provides-Extra: vllm\nProvides-Extra: sglang\n"
                    "Requires-Dist: vllm==0.33.0; extra == 'vllm'\n"
                    "Requires-Dist: sglang==0.5.23; extra == 'sglang'\n",
                )
                wheel.writestr(
                    info + "/entry_points.txt",
                    "[console_scripts]\n"
                    "orbitkv-cache-manager = orbitkv._cache_manager:main\n"
                    "[vllm.general_plugins]\norbitkv = orbitkv.vllm.plugin:register\n"
                    "[sglang.srt.plugins]\norbitkv = orbitkv.sglang.plugin:register\n",
                )
                for license_name in ("LICENSE", "NOTICE"):
                    wheel.writestr(info + "/licenses/" + license_name, "fixture")
    return module, tmp_path


@pytest.mark.parametrize("change", ["none", "missing", "duplicate", "version", "host-library"])
def test_release_checks_each_python_and_architecture_cell(artifacts, change):
    module, directory = artifacts
    first = next(directory.glob("*.whl"))
    if change == "missing":
        first.unlink()
    elif change == "duplicate":
        first.with_name(first.name.replace("2_28", "2_17")).write_bytes(first.read_bytes())
    elif change == "version":
        first.rename(first.with_name(first.name.replace("0.1.0b1", "0.1.0")))
    elif change == "host-library":
        with ZipFile(first, "a") as wheel:
            wheel.writestr("orbitkv.libs/libibverbs-fixture.so.1", b"host-owned")
    if change == "none":
        module.check_release(directory, "0.1.0b1")
    else:
        with pytest.raises(ValueError):
            module.check_release(directory, "0.1.0b1")


@pytest.mark.parametrize("change", ["none", "partial", "digest", "unexpected", "network"])
def test_pypi_receipt_is_exact_and_failed_uploads_are_not_retried(artifacts, monkeypatch, change):
    module, directory = artifacts
    urls = [
        {
            "filename": path.name,
            "digests": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        }
        for path in sorted(directory.glob("*.whl"))
    ]
    if change == "partial":
        urls.pop()
    elif change == "digest":
        urls[0]["digests"]["sha256"] = "0" * 64
    elif change == "unexpected":
        urls.append({"filename": "unexpected.tar.gz", "digests": {"sha256": "0" * 64}})
    calls = []

    def reply(url, timeout):
        calls.append(url)
        assert url == "https://pypi.org/pypi/orbitkv/0.1.0b1/json" and timeout == 30
        if change == "network":
            raise URLError("injected network failure")
        return io.BytesIO(json.dumps({"info": {"version": "0.1.0b1"}, "urls": urls}).encode())

    monkeypatch.setattr(module, "urlopen", reply)
    if change == "none":
        module.check_release(directory, "0.1.0b1", verify_pypi=True)
    else:
        with pytest.raises((ValueError, URLError)):
            module.check_release(directory, "0.1.0b1", verify_pypi=True)
    assert len(calls) == 1
