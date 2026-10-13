"""Verify a staged three-host campaign input manifest without trusting itself."""

from __future__ import annotations

import hashlib
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
    missing = required - entries.keys()
    if missing:
        raise RuntimeError(f"input manifest omits imported helpers: {sorted(missing)}")
    return entries
