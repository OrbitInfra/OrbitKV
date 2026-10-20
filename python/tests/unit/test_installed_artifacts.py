"""A release gate must reject modified or source-backed installed packages."""

import base64
import csv
import hashlib
import json
from importlib.metadata import PathDistribution

import pytest

from tests.support.installed_artifacts import distribution_snapshot, isolated_environment


@pytest.fixture
def installed_distribution(tmp_path):
    package = tmp_path / "sample"
    package.mkdir()
    (package / "__init__.py").write_text("VERSION = '1.0'\n")
    info = tmp_path / "sample-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: sample\nVersion: 1.0\n")
    records = []
    for path in (package / "__init__.py", info / "METADATA"):
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        records.append((str(path.relative_to(tmp_path)), f"sha256={digest}", len(data)))
    (package / "empty").mkdir()
    empty_digest = base64.urlsafe_b64encode(hashlib.sha256(b"").digest()).rstrip(b"=").decode()
    records.append(("sample/empty/", f"sha256={empty_digest}", 0))
    records.append(("sample-1.0.dist-info/RECORD", "", ""))
    with (info / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(records)
    return PathDistribution(info)


@pytest.mark.parametrize(
    "change",
    [
        "unchanged",
        "modified",
        "missing",
        "added",
        "added_data",
        "generated_bytecode",
        "cache_payload",
        "editable",
        "unhashed",
        "unhashed_data",
        "foreign_record",
        "missing_record",
        "missing_directory",
    ],
)
def test_installed_record_integrity(installed_distribution, change):
    package = installed_distribution.locate_file("sample")
    expected = distribution_snapshot(installed_distribution, "sample")
    if change == "modified":
        (package / "__init__.py").write_text("VERSION = 'patched'\n")
    elif change == "missing":
        (package / "__init__.py").unlink()
    elif change == "added":
        (package / "runtime_patch.py").write_text("patched = True\n")
    elif change == "added_data":
        (package / "runtime_config.json").write_text("unrecorded configuration")
    elif change in {"generated_bytecode", "cache_payload"}:
        cache = package / "__pycache__"
        cache.mkdir()
        filename = "__init__.cpython-311.pyc" if change == "generated_bytecode" else "config.json"
        (cache / filename).write_bytes(b"generated cache")
    elif change == "editable":
        installed_distribution.locate_file("sample-1.0.dist-info/direct_url.json").write_text(
            json.dumps({"dir_info": {"editable": True}})
        )
    elif change == "unhashed":
        record = installed_distribution.locate_file("sample-1.0.dist-info/RECORD")
        rows = list(csv.reader(record.read_text().splitlines()))
        rows[0][1] = ""
        with record.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
    elif change in {"unhashed_data", "foreign_record"}:
        filename = "runtime_config.json" if change == "unhashed_data" else "RECORD"
        (package / filename).write_text("unverified contents")
        record = installed_distribution.locate_file("sample-1.0.dist-info/RECORD")
        with record.open("a", newline="") as stream:
            csv.writer(stream).writerow((f"sample/{filename}", "", ""))
    elif change == "missing_record":
        installed_distribution.locate_file("sample-1.0.dist-info/RECORD").unlink()
    elif change == "missing_directory":
        (package / "empty").rmdir()
    if change in {"unchanged", "generated_bytecode"}:
        assert distribution_snapshot(installed_distribution, "sample") == expected
    else:
        with pytest.raises(AssertionError):
            distribution_snapshot(installed_distribution, "sample")


def test_release_environment_excludes_source_and_native_overrides(tmp_path):
    native = tmp_path / "staged"
    native.mkdir()
    (native / "libtent_shared.so").touch()
    env = isolated_environment(
        {
            "PATH": "/usr/bin",
            "PYTHONPATH": "/source/python",
            "PYTHONHOME": "/source/venv",
            "ORBITKV_CACHE_MANAGER_BINARY": "/source/manager",
            "ORBITKV_MODEL_FINGERPRINT": "inherited",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "LD_LIBRARY_PATH": f"/source/.orbitkv/mooncake:{native}:/usr/local/cuda/lib64",
        },
        tmp_path,
    )
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert "ORBITKV_CACHE_MANAGER_BINARY" not in env
    assert "ORBITKV_MODEL_FINGERPRINT" not in env
    assert env["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"


@pytest.mark.parametrize("mode", [0o755, 0o644])
def test_orbitkv_record_guard_requires_executable_manager(installed_distribution, mode):
    package = installed_distribution.locate_file("sample")
    orbitkv = package.with_name("orbitkv")
    package.rename(orbitkv)
    manager = orbitkv / "orbitkv-cache-manager-py"
    manager.write_bytes(b"frozen manager bytes")
    manager.chmod(mode)
    record = installed_distribution.locate_file("sample-1.0.dist-info/RECORD")
    rows = list(csv.reader(record.read_text().splitlines()))
    for row in rows:
        if row[0].startswith("sample/"):
            row[0] = "orbitkv/" + row[0].removeprefix("sample/")
    data = manager.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    rows.insert(-1, ("orbitkv/orbitkv-cache-manager-py", f"sha256={digest}", len(data)))
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    if mode & 0o111:
        snapshot = distribution_snapshot(installed_distribution, "orbitkv")
        assert snapshot["executables"] == {str(manager): manager.stat().st_mode}
    else:
        with pytest.raises(AssertionError, match="execute"):
            distribution_snapshot(installed_distribution, "orbitkv")
