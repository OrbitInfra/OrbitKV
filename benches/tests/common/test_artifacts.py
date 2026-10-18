"""Exercise the output boundary and the tracked-file guard without a GPU runtime."""

import os
import subprocess
import sys
from argparse import ArgumentTypeError

import pytest

from benches.common.artifacts import ROOT, external_path


def test_external_output_resolves_paths_and_rejects_checkout_symlinks(tmp_path):
    assert external_path(str(tmp_path / "run")) == tmp_path / "run"
    alias = tmp_path / "checkout"
    alias.symlink_to(ROOT, target_is_directory=True)
    for path in (ROOT, ROOT / "benches/results/new", alias / "run"):
        with pytest.raises(ArgumentTypeError, match="outside the source checkout"):
            external_path(str(path))


def test_guard_checks_git_index_including_force_added_output_and_allows_fixtures(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    def check():
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/check-experiment-output.py")],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

    git("init")
    (tmp_path / ".gitignore").write_text("/benches/results/\n/results/\n")
    for name in ("benches/tests/fixtures/results.json", "src/result.rs", "docs/results.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("deterministic input\n")
    git("add", ".")
    assert check().returncode == 0
    for name in (
        "benches/results/forced.csv",
        "benches/runs/raw.log",
        "results/raw.json",
        "runs/failed.txt",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("experiment\n")
        git("add", "-f", name)
        result = check()
        assert result.returncode == 1
        assert name in result.stdout
        git("rm", "--cached", name)
    assert check().returncode == 0


def test_serving_script_requires_external_result_dir_before_launch(tmp_path):
    env = {**os.environ, "BASE_URL": "http://unused", "MODEL": "unused", "LABEL": "test"}
    env.pop("RESULT_DIR", None)
    script = str(ROOT / "benches/serving/serving.sh")
    missing = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "set RESULT_DIR" in missing.stderr
    env["RESULT_DIR"] = str(ROOT / "benches/results/rejected")
    rejected = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert "outside the source checkout" in rejected.stderr
    command = tmp_path / "vllm"
    command.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$RESULT_DIR/arguments.txt"\n')
    command.chmod(0o755)
    env.update(RESULT_DIR=str(tmp_path / "serving"), VLLM_BIN=str(command), LENGTHS="1024")
    subprocess.run(["bash", script], env=env, check=True, capture_output=True)
    assert (tmp_path / "serving/arguments.txt").is_file()
