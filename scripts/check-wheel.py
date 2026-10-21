"""Check the installable wheel, including files staged outside Cargo's build."""

import argparse
import configparser
import email
import os
import stat
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile


def check_wheel(path: Path, version: str | None = None) -> None:
    expected_name = "orbitkv-llm"
    required_files = {
        "orbitkv/__init__.py",
        "orbitkv/orbitkv.pyi",
        "orbitkv/client/gpu.py",
        "orbitkv/vllm/plugin.py",
        "orbitkv/vllm/connector.py",
        "orbitkv/vllm/config.py",
        "orbitkv/vllm/layout.py",
        "orbitkv/vllm/metadata.py",
        "orbitkv/vllm/metrics.py",
        "orbitkv/sglang/plugin.py",
        "orbitkv/sglang/linker.py",
        "orbitkv/sglang/config.py",
        "orbitkv/sglang/layout.py",
        "orbitkv/orbitkv-cache-manager-py",
        "orbitkv/libtent_shared.so",
        "orbitkv/libmooncake_common.so",
        "orbitkv/libasio.so",
    }
    with ZipFile(path) as wheel:
        files = set(wheel.namelist())
        missing = required_files - files
        if missing:
            raise ValueError(f"missing wheel files: {', '.join(sorted(missing))}")
        removed_files = {
            "orbitkv/vllm/transport.py",
            "orbitkv/vllm/runtime.py",
            "orbitkv/sglang/pd.py",
            "orbitkv/sglang/completion.py",
            "orbitkv/libtransfer_engine.so",
            "orbitkv/orbitkv-metaserver-py",
            "orbitkv/orbitkv-catalog-py",
            "orbitkv/_metaserver.py",
            "orbitkv/vllm_plugin.py",
            "orbitkv/sglang/storage.py",
            "orbitkv/sglang/hicache.py",
            "orbitkv/vllm/common.py",
            "orbitkv/vllm/connector_metrics.py",
            "orbitkv/sglang/pools.py",
            "orbitkv/ipc_wrapper.py",
        }
        removed_prefixes = (
            "orbitkv/connector/",
            "orbitkv/pd_connector/",
            "orbitkv/nixl_connector/",
            "orbitkv/vllm/nixl/",
            "orbitkv/vllm/pd/",
            "tests/",
            "benches/",
        )
        unexpected = (removed_files & files) | {
            name for name in files if name.startswith(removed_prefixes)
        }
        if unexpected:
            raise ValueError(
                f"wheel contains removed adapter paths: {', '.join(sorted(unexpected))}"
            )
        if any("/__pycache__/" in name or name.endswith(".pyc") for name in files):
            raise ValueError("wheel contains Python build caches")
        if not any(
            name.startswith("orbitkv/orbitkv.") and name.endswith(".so")
            for name in files
        ):
            raise ValueError("wheel is missing the Rust extension")

        metadata_files = [
            name for name in files if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError("wheel must have exactly one METADATA file")
        metadata = email.message_from_bytes(wheel.read(metadata_files[0]))
        if metadata["Name"] != expected_name:
            raise ValueError(f"expected {expected_name}, got {metadata['Name']}")
        if version is not None and metadata["Version"] != version:
            raise ValueError(f"expected version {version}, got {metadata['Version']}")
        extras = set(metadata.get_all("Provides-Extra", []))
        if extras != {"vllm", "sglang"}:
            raise ValueError(f"wheel should expose only engine extras: {extras}")
        requirements = metadata.get_all("Requires-Dist", [])
        for engine in ("vllm", "sglang"):
            if not any(
                req.startswith(f"{engine}==") and f"extra == '{engine}'" in req
                for req in requirements
            ):
                raise ValueError(f"missing exact {engine} release dependency")
        if any("orbitkv-llm[test]" in req for req in requirements):
            raise ValueError("wheel has a self-dependency on its test extra")

        entry_points = metadata_files[0].replace("METADATA", "entry_points.txt")
        if entry_points not in files:
            raise ValueError("wheel is missing console and engine entry points")
        entries = configparser.ConfigParser(interpolation=None)
        entries.read_string(wheel.read(entry_points).decode())
        expected_entries = {
            ("console_scripts", "orbitkv-cache-manager"): "orbitkv._cache_manager:main",
            ("vllm.general_plugins", "orbitkv"): "orbitkv.vllm.plugin:register",
            ("sglang.srt.plugins", "orbitkv"): "orbitkv.sglang.plugin:register",
        }
        for (group, name), target in expected_entries.items():
            if entries.get(group, name, fallback=None) != target:
                raise ValueError(f"missing entry point: {group}/{name} -> {target}")

        for license_file in ("LICENSE", "NOTICE"):
            if not any(
                name.endswith(f".dist-info/licenses/{license_file}") for name in files
            ):
                raise ValueError(f"wheel is missing {license_file}")

        for binary in ("orbitkv-cache-manager-py",):
            mode = wheel.getinfo(f"orbitkv/{binary}").external_attr >> 16
            if not mode & stat.S_IXUSR:
                raise ValueError(f"{binary} is not executable")
        corrupt = wheel.testzip()
        if corrupt:
            raise ValueError(f"corrupt wheel member: {corrupt}")
    print(f"Wheel validated: {path.name} ({metadata['Version']})")


def check_install(path: Path) -> None:
    distribution = "orbitkv-llm"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    import_check = (
        "import importlib.metadata as metadata; import orbitkv; "
        "import orbitkv.vllm.plugin; import orbitkv.sglang.plugin; "
        "from orbitkv.client import CacheManagerClient; "
        f"assert orbitkv.__version__ == metadata.version({distribution!r}); "
        "assert orbitkv.ChannelProbeClient and orbitkv.CacheManagerClient is CacheManagerClient"
    )
    with tempfile.TemporaryDirectory(prefix="orbitkv-wheel-") as directory:
        root = Path(directory)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = root / "venv/bin/python"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                str(path.resolve()),
            ],
            check=True,
            cwd=root,
            env=env,
        )
        subprocess.run(
            [str(python), "-c", import_check],
            check=True,
            cwd=root,
            env=env,
        )
    print(f"Isolated native import passed: {distribution}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--version", help="Expected release version")
    parser.add_argument("--install-smoke", action="store_true")
    args = parser.parse_args()
    try:
        check_wheel(args.wheel, args.version)
        if args.install_smoke:
            check_install(args.wheel)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Wheel validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
