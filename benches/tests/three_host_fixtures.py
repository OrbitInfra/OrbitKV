"""Test-owned source staging and manifests for the three-host harness controls."""

import hashlib
import shutil
from pathlib import Path

from benches.three_host_etcd import integrity

_COMMON_HELPERS = (
    "campaign.py",
    "consumer_agent.py",
    "environment.py",
    "integrity.py",
    "launch_agent.py",
    "network_probe_agent.py",
    "node_agent.py",
    "proxy_readiness.py",
    "resource_lock_agent.py",
    "validate_inputs.py",
)


def stage_helpers(inputs: Path, *, process_only: bool = False, stub: bool = False) -> None:
    names = (*_COMMON_HELPERS, "process_qualification.py") if process_only else _COMMON_HELPERS
    source = Path(integrity.__file__).parent
    for name in names:
        if stub:
            (inputs / name).write_text("\n")
        else:
            shutil.copy2(source / name, inputs / name)


def freeze_inputs(root: Path) -> str:
    inputs = root / "inputs"
    manifest = inputs / "SHA256SUMS"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(inputs.iterdir())
            if path != manifest
        )
    )
    return hashlib.sha256(manifest.read_bytes()).hexdigest()
