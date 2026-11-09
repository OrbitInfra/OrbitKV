"""CI must bypass an unavailable cache without hiding rustc failure."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/configure-rustc-wrapper.sh"


def executable(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def run_probe(tmp_path: Path, wrapper_status: int | None, rustc_status: int):
    tools = tmp_path / "bin"
    tools.mkdir()
    wrapper = tools / "cache"
    if wrapper_status is not None:
        executable(wrapper, f"exit {wrapper_status}")
    executable(tools / "rustc", f"printf 'rustc-probe\\n'; exit {rustc_status}")
    github_env = tmp_path / "github-env"
    environment = {
        **os.environ,
        "PATH": f"{tools}:{os.environ['PATH']}",
        "RUSTC_WRAPPER": str(wrapper),
        "GITHUB_ENV": str(github_env),
        "RUNNER_TEMP": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", SCRIPT],
        env=environment,
        text=True,
        capture_output=True,
    )
    return result, github_env


def test_available_cache_remains_selected(tmp_path):
    result, github_env = run_probe(tmp_path, wrapper_status=0, rustc_status=9)
    assert result.returncode == 0
    assert not github_env.exists()


@pytest.mark.parametrize("wrapper_status", [2, None], ids=["broken", "not-installed"])
def test_cache_failure_selects_direct_rustc(tmp_path, wrapper_status):
    result, github_env = run_probe(tmp_path, wrapper_status=wrapper_status, rustc_status=0)
    assert result.returncode == 0
    assert "using direct rustc" in result.stdout
    assert github_env.read_text() == "RUSTC_WRAPPER=\nSCCACHE_GHA_ENABLED=false\n"


@pytest.mark.parametrize("wrapper_status", [2, None], ids=["broken", "not-installed"])
def test_direct_rustc_failure_remains_fatal(tmp_path, wrapper_status):
    result, github_env = run_probe(tmp_path, wrapper_status=wrapper_status, rustc_status=9)
    assert result.returncode == 9
    assert github_env.is_file()
