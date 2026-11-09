"""Transient download errors may retry; exhausted or invalid installs stay fatal."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.unit.test_ci_rustc_wrapper import executable

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/install-protoc.sh"


@pytest.mark.parametrize(
    "failures,version,success,attempts",
    [(0, "27.5", True, 1), (2, "27.5", True, 3), (3, "27.5", False, 3), (0, "27.4", False, 1)],
)
def test_pinned_protoc_retry_is_bounded_and_fatal(tmp_path, failures, version, success, attempts):
    tools = tmp_path / "bin"
    tools.mkdir()
    executable(tools / "uname", "echo 'Linux x86_64'")
    executable(tools / "sleep", "exit 0")
    executable(
        tools / "curl",
        """
echo "$*" >> "$RUNNER_TEMP/calls"
count=$(wc -l < "$RUNNER_TEMP/calls")
test "$count" -gt "$FAILURES"
""",
    )
    # Network and archive extraction are fault seams; the installer itself is
    # run unchanged and must verify the actual selected executable's version.
    executable(
        tools / "unzip",
        """
mkdir -p "$RUNNER_TEMP/protoc-27.5/bin"
printf '#!/bin/sh\necho "libprotoc %s"\n' "$PROBE_VERSION" > "$RUNNER_TEMP/protoc-27.5/bin/protoc"
chmod +x "$RUNNER_TEMP/protoc-27.5/bin/protoc"
""",
    )
    github_path = tmp_path / "github-path"
    result = subprocess.run(
        ["bash", SCRIPT],
        text=True,
        capture_output=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_PATH": str(github_path),
            "FAILURES": str(failures),
            "PROBE_VERSION": version,
        },
    )
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == attempts
    assert all("v27.5/protoc-27.5-linux-x86_64.zip" in call for call in calls)
    assert (result.returncode == 0) is success, result.stderr
    assert github_path.exists() is success
    if success:
        assert github_path.read_text() == str(tmp_path / "protoc-27.5/bin") + "\n"
    elif failures == 3:
        assert "failed after 3 attempts" in result.stderr
    else:
        assert "does not match" in result.stderr
