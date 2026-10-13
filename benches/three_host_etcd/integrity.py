"""Verify a staged three-host campaign input manifest without trusting itself."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import sysconfig
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(base: Path, manifest: Path, approved_manifest_sha256: str) -> dict[str, str]:
    actual_manifest_sha256 = sha256(manifest)
    if actual_manifest_sha256 != approved_manifest_sha256:
        raise RuntimeError(
            "input manifest differs from the independently approved digest: "
            f"{actual_manifest_sha256} != {approved_manifest_sha256}"
        )
    entries: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text().splitlines(), start=1):
        digest, separator, name = line.partition("  ")
        relative = PurePosixPath(name)
        if (
            len(digest) != 64
            or separator != "  "
            or relative.is_absolute()
            or ".." in relative.parts
            or name in entries
        ):
            raise RuntimeError(f"invalid input manifest line {number}: {line!r}")
        path = base / Path(*relative.parts)
        resolved_base = base.resolve(strict=True)
        if not path.resolve(strict=True).is_relative_to(resolved_base):
            raise RuntimeError(f"manifest input escapes frozen base: {name}")
        if not path.is_file():
            raise RuntimeError(f"manifest input is missing: {name}")
        actual = sha256(path)
        if actual != digest:
            raise RuntimeError(f"manifest input differs: {name}: {actual} != {digest}")
        entries[name] = digest
    return entries


def verify_inputs(root: Path, approved_manifest_sha256: str) -> dict[str, str]:
    entries = verify_manifest(root / "inputs", root / "inputs/SHA256SUMS", approved_manifest_sha256)
    required = {
        "campaign.py",
        "consumer_agent.py",
        "environment.py",
        "integrity.py",
        "launch_agent.py",
        "network_probe_agent.py",
        "node_agent.py",
        "proxy_readiness.py",
        "resource_lock_agent.py",
        "settings.json",
        "validate_inputs.py",
    }
    if (root / "inputs/process_qualification.py").exists():
        required.add("process_qualification.py")
    settings = json.loads((root / "inputs/settings.json").read_text())
    if "process_qualification" in settings:
        required.update(f"{node['name']}.runtime.json" for node in settings["nodes"])
    missing = required - entries.keys()
    if missing:
        raise RuntimeError(f"input manifest omits imported helpers: {sorted(missing)}")
    return entries


PROCESS_MODULES = ("requests", "urllib3", "certifi", "charset_normalizer", "idna")


def process_runtime_snapshot() -> dict:
    """Capture the interpreter and consumed HTTP runtime before freezing inputs."""
    executable = Path("/proc/self/exe").resolve(strict=True)
    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    files = {str(executable): sha256(executable)}
    links = {}
    origins = {}
    roots = [stdlib]
    for name in PROCESS_MODULES:
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve(strict=True)
        origins[name] = str(origin)
        roots.append(origin.parent)
    for root in roots:
        for directory, children, names in os.walk(root):
            children[:] = [
                name
                for name in children
                if name != "__pycache__"
                and not (root == stdlib and name in {"site-packages", "dist-packages"})
            ]
            for name in names:
                path = Path(directory) / name
                if path.is_file():
                    files[str(path)] = sha256(path)
                    if path.is_symlink():
                        links[str(path)] = str(path.resolve(strict=True))
    mappings = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            if not path.is_file():
                raise RuntimeError(f"process runtime mapped file is unavailable: {path}")
            mappings.add(str(path.resolve(strict=True)))
            files[str(path.resolve(strict=True))] = sha256(path)
    return {
        "python": str(executable),
        "version": sys.version,
        "stdlib": str(stdlib),
        "origins": origins,
        "files": files,
        "links": links,
        "mapped_files": sorted(mappings),
    }


def process_python(root: Path, node_name: str) -> list[str]:
    snapshot = json.loads((root / "inputs" / f"{node_name}.runtime.json").read_text())
    return [
        snapshot["python"],
        "-B",
        "-E",
        "-s",
        "-X",
        f"pycache_prefix={root / '.runtime-bytecode'}",
    ]


def verify_process_runtime(root: Path, settings: dict, node_name: str | None = None) -> dict:
    """Reject a changed or shadowed runtime before this process owns resources."""
    if "process_qualification" not in settings:
        return {}
    bytecode = root / ".runtime-bytecode"
    if (
        sys.pycache_prefix != str(bytecode)
        or not sys.flags.dont_write_bytecode
        or not sys.flags.ignore_environment
        or not sys.flags.no_user_site
        or bytecode.exists()
        or bytecode.is_symlink()
    ):
        raise RuntimeError("process runtime bytecode/environment isolation differs")
    if node_name is None:
        import socket

        matches = [node for node in settings["nodes"] if node["hostname"] == socket.gethostname()]
        if len(matches) != 1:
            raise RuntimeError("process runtime hostname is not uniquely frozen")
        node_name = matches[0]["name"]
    snapshot = json.loads((root / "inputs" / f"{node_name}.runtime.json").read_text())
    if (
        set(snapshot)
        != {"python", "version", "stdlib", "origins", "files", "links", "mapped_files"}
        or set(snapshot["origins"]) != set(PROCESS_MODULES)
        or not snapshot["files"]
        or not snapshot["mapped_files"]
        or str(Path("/proc/self/exe").resolve(strict=True)) != snapshot["python"]
        or sys.version != snapshot["version"]
        or str(Path(sysconfig.get_path("stdlib")).resolve(strict=True)) != snapshot["stdlib"]
    ):
        raise RuntimeError("process interpreter/runtime contract differs")
    for name, origin in snapshot["origins"].items():
        module = importlib.import_module(name)
        if str(Path(module.__file__).resolve(strict=True)) != origin:
            raise RuntimeError(f"process runtime import origin differs: {name}")
    current = process_runtime_snapshot()
    expected_files = set(snapshot["files"]) - set(snapshot["mapped_files"])
    current_files = set(current["files"]) - set(snapshot["mapped_files"])
    if current_files != expected_files:
        raise RuntimeError("process runtime file inventory or digest differs")
    for name, digest in snapshot["files"].items():
        if not Path(name).is_absolute() or sha256(Path(name)) != digest:
            raise RuntimeError("process runtime file inventory or digest differs")
    if current["links"] != snapshot["links"]:
        raise RuntimeError("process runtime link differs")
    observed = set(current["mapped_files"])
    if not observed <= set(snapshot["mapped_files"]):
        raise RuntimeError(
            f"unfrozen process runtime mapping: {sorted(observed - set(snapshot['mapped_files']))}"
        )
    return {
        "node": node_name,
        "python": snapshot["python"],
        "files": len(snapshot["files"]),
        "origins": snapshot["origins"],
        "mapped_files": sorted(observed),
    }
