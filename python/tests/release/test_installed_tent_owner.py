"""Installed Python allocation owners with the matching CPU TENT fault fixture.

This explicit release/integration gate supplies no CUDA or RDMA qualification.
Each case owns one fresh subprocess; failed owners remain held until that exit.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.release_smoke]


@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "invalid",
        "retry",
        "gc-after-close-error",
        "bulk-register-error",
        "native-partial",
        "borrowed-error",
    ],
)
def test_installed_tent_allocation_owner(case):
    control = os.environ.get("ORBITKV_TENT_OWNER_CONTROL_LIBRARY")
    if not control:
        pytest.skip("requires the frozen CPU-only TENT and matching owner control bundle")
    import orbitkv.orbitkv as native

    extension = Path(native.__file__).resolve(strict=True)
    if "site-packages" not in extension.parts:
        pytest.fail(f"requires an installed extension: {extension}")
    helper = Path(__file__).resolve().parents[1] / "support" / "tent_memory_owner.py"
    environment = os.environ.copy()
    environment["MC_FORCE_TCP"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", str(helper), str(extension.parent.parent), case],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (case, result.stdout, result.stderr)
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1 and rows[0]["case"] == case and rows[0]["passed"]
    print(json.dumps(rows[0], sort_keys=True))
