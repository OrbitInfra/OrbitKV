"""Transient download errors may retry; exhausted or invalid installs stay fatal."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.unit.test_ci_rustc_wrapper import executable

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/install-protoc.sh"


@pytest.mark.parametrize(
    "arch,failures,version,hash_ok,success,attempts",
    [
        ("x86_64", 0, "27.5", True, True, 1),
        ("aarch64", 0, "27.5", True, True, 1),
        ("x86_64", 2, "27.5", True, True, 3),
        ("x86_64", 3, "27.5", True, False, 3),
        ("x86_64", 0, "27.4", True, False, 1),
        ("x86_64", 0, "27.5", False, False, 1),
    ],
)
def test_pinned_protoc_retry_is_bounded_and_fatal(
    tmp_path, arch, failures, version, hash_ok, success, attempts
):
    tools = tmp_path / "bin"
    tools.mkdir()
    executable(tools / "uname", f"echo 'Linux {arch}'")
    executable(tools / "sleep", "exit 0")
    executable(
        tools / "curl",
        """
echo "$*" >> "$RUNNER_TEMP/calls"
count=$(wc -l < "$RUNNER_TEMP/calls")
test "$count" -gt "$FAILURES"
""",
    )
    # Download, checksum and extraction are fault seams. The unchanged installer
    # must reject a failed hash before extraction, then verify the binary version.
    executable(
        tools / "sha256sum",
        """
test "$*" = '--check --strict'
cat > "$RUNNER_TEMP/checksum-input"
test "$HASH_OK" = 1
""",
    )
    executable(
        tools / "unzip",
        """
mkdir -p "$RUNNER_TEMP/protoc-27.5/bin"
touch "$RUNNER_TEMP/extracted"
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
            "HASH_OK": str(int(hash_ok)),
        },
    )
    calls = (tmp_path / "calls").read_text().splitlines()
    assert len(calls) == attempts
    platform = "aarch_64" if arch == "aarch64" else arch
    assert all(f"v27.5/protoc-27.5-linux-{platform}.zip" in call for call in calls)
    assert (result.returncode == 0) is success, result.stderr
    assert github_path.exists() is success
    assert (tmp_path / "extracted").exists() is (failures < 3 and hash_ok)
    if failures < 3:
        digest, archive = (tmp_path / "checksum-input").read_text().strip().split(maxsplit=1)
        assert len(bytes.fromhex(digest)) == 32
        assert Path(archive).name == "protoc.zip"
    if success:
        assert github_path.read_text() == str(tmp_path / "protoc-27.5/bin") + "\n"
    elif failures == 3:
        assert "failed after 3 attempts" in result.stderr
    elif not hash_ok:
        assert "SHA-256 mismatch" in result.stderr
    else:
        assert "does not match" in result.stderr
