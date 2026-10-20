"""Installed host-buffer TCP gate; independent of GPU or engine serving qualification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.release_smoke, pytest.mark.integration]


@pytest.mark.parametrize("case", ["tcp-payload", "tcp-payload-register-error"])
def test_installed_tcp_payload_and_normal_registration(case):
    if case.endswith("register-error") and not os.environ.get("ORBITKV_TENT_OWNER_CONTROL_LIBRARY"):
        pytest.skip("requires the matching CPU-only registration refusal control")
    import orbitkv.orbitkv as native

    extension = Path(native.__file__).resolve(strict=True)
    if "site-packages" not in extension.parts or "target" in extension.parts:
        pytest.fail(f"TCP artifact gate requires an installed extension: {extension}")
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
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1 and rows[0]["case"] == case, (result.stdout, result.stderr)
    if case == "tcp-payload":
        assert result.returncode == 0 and rows[0]["passed"], (result.stdout, result.stderr)
    else:
        assert result.returncode == 1 and not rows[0]["passed"], (result.stdout, result.stderr)
        assert rows[0]["attempted_regions"] == 2
        assert "register memory failed" in rows[0]["first_error"]
        assert rows[0]["native_counts"] == {
            "registers": 2,
            "unregisters": 0,
            "source_closes": 0,
            "destroys": 0,
        }
    print(json.dumps(rows[0], sort_keys=True))
