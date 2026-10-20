"""Installed-package provenance for release gates; runnable in isolated Python."""

import base64
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import stat
import sys
import sysconfig
from pathlib import Path

ENGINE_VERSIONS = {"vllm": "0.31.0", "sglang": "0.5.21"}
ENGINE_COMMITS = {
    "vllm": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
    "sglang": "e00930c5489053f26d86b179cee0d087f846acbb",
}
NATIVE_LIBRARIES = {"libtent_shared.so", "libmooncake_common.so", "libasio.so"}
EXECUTABLE_PACKAGE_FILES = {"orbitkv": ("orbitkv-cache-manager-py",)}


def isolated_environment(environment: dict[str, str], directory: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in environment.items()
        if not key.startswith(("PYTHON", "ORBITKV_"))
    }
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        part
        for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if part
        and "/.orbitkv/" not in part
        and not part.endswith("/python/orbitkv")
        and not any((Path(part) / name).exists() for name in NATIVE_LIBRARIES)
    )
    env.update(
        ORBITKV_CACHE_SCOPE=directory.name,
        MC_FORCE_TCP="1",
        VLLM_USE_V2_MODEL_RUNNER="0",
        VLLM_BATCH_INVARIANT="1",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        TORCHINDUCTOR_CACHE_DIR=str(directory / "inductor"),
        TRITON_CACHE_DIR=str(directory / "triton"),
        VLLM_CACHE_ROOT=str(directory / "vllm"),
    )
    return env


def distribution_snapshot(distribution, package: str) -> dict:
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    assert not direct_url.get("dir_info", {}).get("editable", False), direct_url
    record = distribution.read_text("RECORD")
    assert record, f"{distribution.metadata['Name']} has no installed RECORD"
    rows = list(csv.reader(record.splitlines()))
    assert rows, f"{distribution.metadata['Name']} has an empty installed RECORD"
    record_path = (Path(distribution._path) / "RECORD").resolve()
    hashes = {}
    recorded = set()
    for relative, checksum, size in rows:
        path = Path(distribution.locate_file(relative)).resolve()
        recorded.add(path)
        assert checksum or path == record_path, f"Installed file has no RECORD hash: {path}"
        if relative.endswith("/"):
            assert path.is_dir(), f"Missing installed directory: {path}"
            assert size == "0", f"Nonempty directory RECORD: {relative}"
            data = b""
        else:
            assert path.is_file(), f"Missing installed file: {path}"
            data = path.read_bytes()
        if checksum:
            algorithm, expected = checksum.split("=", 1)
            actual = base64.urlsafe_b64encode(hashlib.new(algorithm, data).digest())
            assert actual.rstrip(b"=").decode() == expected, (
                f"Installed file differs from RECORD: {path}"
            )
        if not relative.endswith("/"):
            hashes[str(path)] = hashlib.sha256(data).hexdigest()
    package_root = Path(distribution.locate_file(package)).resolve()
    assert package_root.is_dir(), package_root
    executable_files = {}
    for relative in EXECUTABLE_PACKAGE_FILES.get(package, ()):
        path = package_root / relative
        metadata = path.stat()
        assert path.resolve() in recorded, f"Installed executable is outside RECORD: {path}"
        assert stat.S_ISREG(metadata.st_mode), f"Installed executable is not regular: {path}"
        assert metadata.st_mode & 0o111, f"Installed executable has no execute bit: {path}"
        assert os.access(path, os.X_OK), f"Installed executable cannot execute: {path}"
        executable_files[str(path.resolve())] = metadata.st_mode
    unrecorded = [
        str(path)
        for path in package_root.rglob("*")
        if path.is_file()
        and path.resolve() not in recorded
        and not (path.suffix == ".pyc" and "__pycache__" in path.relative_to(package_root).parts)
    ]
    assert not unrecorded, f"Unrecorded runtime files: {unrecorded}"
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "package_root": str(package_root),
        "files": hashes,
        "executables": executable_files,
    }


def installation_snapshot(engine: str, initialize_transfer: bool = False) -> dict:
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    owners = metadata.packages_distributions().get("orbitkv", [])
    assert len(owners) == 1, f"Expected one OrbitKV distribution, got {owners}"
    distribution = metadata.distribution(owners[0])
    assert distribution.metadata["Name"] in {"orbitkv-llm", "orbitkv-llm-cu13"}
    engine_distribution = metadata.distribution(engine)
    assert engine_distribution.version == ENGINE_VERSIONS[engine], engine_distribution.version
    snapshots = {}
    for package, installed in (("orbitkv", distribution), (engine, engine_distribution)):
        spec = importlib.util.find_spec(package)
        assert spec and spec.origin, f"Missing installed package: {package}"
        origin = Path(spec.origin).resolve()
        assert origin.is_relative_to(purelib), f"Package outside this environment: {origin}"
        snapshot = distribution_snapshot(installed, package)
        assert origin.parent == Path(snapshot["package_root"]), origin
        snapshots[package] = snapshot
    for group, target in (
        ("vllm.general_plugins", "orbitkv.vllm.plugin:register"),
        ("sglang.srt.plugins", "orbitkv.sglang.plugin:register"),
    ):
        entries = list(metadata.entry_points(group=group, name="orbitkv"))
        assert len(entries) == 1 and entries[0].value == target, entries
        assert entries[0].dist.metadata["Name"] == distribution.metadata["Name"], entries
    import orbitkv

    for plugin in ("orbitkv.vllm.plugin", "orbitkv.sglang.plugin"):
        importlib.import_module(plugin)
    assert "torch" not in sys.modules and "orbitkv.orbitkv" not in sys.modules, (
        "Plugin discovery initialized the GPU/native runtime"
    )
    assert orbitkv.__version__ == distribution.version
    loaded = []
    if initialize_transfer:
        transfer = orbitkv.MooncakeTransferEngine(bind_host="127.0.0.1")
        loaded = sorted(
            {
                str(Path(line.split()[-1]).resolve())
                for line in Path("/proc/self/maps").read_text().splitlines()
                if Path(line.split()[-1]).name in NATIVE_LIBRARIES
            }
        )
        assert {Path(path).name for path in loaded} == NATIVE_LIBRARIES, loaded
        assert all(Path(path).parent == Path(orbitkv.__file__).parent for path in loaded), loaded
        del transfer
    return {
        "distributions": snapshots,
        "native_libraries": loaded,
        "python": sys.executable,
        "python_libdir": sysconfig.get_config_var("LIBDIR"),
        "pythonhome": os.environ.get("PYTHONHOME"),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    }


if __name__ == "__main__":
    print(json.dumps(installation_snapshot(sys.argv[1], "--native" in sys.argv[2:])))
