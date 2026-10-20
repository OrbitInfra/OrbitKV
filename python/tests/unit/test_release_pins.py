"""Release upgrades must align installable extras, gate inputs and gitlinks."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def release_checker(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/check-versions.py"
    spec = importlib.util.spec_from_file_location("check_release_versions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "python/tests/support").mkdir(parents=True)
    versions = {"vllm": "0.31.0", "sglang": "0.5.22"}
    commits = {"vllm": "a" * 40, "sglang": "b" * 40}
    requirements = {name: f"{name}=={version}" for name, version in versions.items()}
    requirements["sglang"] += "; python_version >= '3.11'"
    (tmp_path / "python/pyproject.toml").write_text(
        "[project.optional-dependencies]\n"
        + "\n".join(f"{name} = {json.dumps([value])}" for name, value in requirements.items())
    )
    helper = tmp_path / "python/tests/support/installed_artifacts.py"
    helper.write_text(f"ENGINE_VERSIONS = {versions!r}\nENGINE_COMMITS = {commits!r}\n")
    entries = {name: f"160000 {commit} 0\tthird-party/{name}\n" for name, commit in commits.items()}
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda args, **_: entries[args[-1].split("/")[-1]]
    )
    return module, tmp_path, helper, entries


@pytest.mark.parametrize("layout", ["ordinary", "multiline"])
def test_aligned_release_inputs_pass(release_checker, layout):
    module, root, _, _ = release_checker
    if layout == "multiline":
        (root / "python/pyproject.toml").write_text(
            "[project.optional-dependencies]\n"
            "vllm = [\n  'vllm==0.31.0', # official release\n]\n"
            "sglang = [\"sglang==0.5.22; python_version >= '3.11'\"]\n"
        )
    module.check_engine_releases(root)


@pytest.mark.parametrize(
    "drift",
    [
        "extra",
        "missing-marker",
        "wrong-marker",
        "wrong-table",
        "gitlink",
        "merge-stage",
        "commit",
        "missing-engine",
    ],
)
def test_release_input_drift_is_rejected(release_checker, drift):
    module, root, helper, entries = release_checker
    if drift == "extra":
        path = root / "python/pyproject.toml"
        path.write_text(path.read_text().replace("vllm==0.31.0", "vllm==0.30.0"))
    elif drift in ("missing-marker", "wrong-marker"):
        path = root / "python/pyproject.toml"
        marker = "" if drift == "missing-marker" else "; python_version < '3.11'"
        path.write_text(path.read_text().replace("; python_version >= '3.11'", marker))
    elif drift == "wrong-table":
        path = root / "python/pyproject.toml"
        path.write_text(path.read_text().replace("[project.optional-dependencies]", "[tool.other]"))
    elif drift == "gitlink":
        entries["sglang"] = f"160000 {'c' * 40} 0\tthird-party/sglang\n"
    elif drift == "merge-stage":
        entries["sglang"] = entries["sglang"].replace(" 0\t", " 1\t")
    elif drift == "commit":
        helper.write_text(helper.read_text().replace("a" * 40, "invalid"))
    else:
        helper.write_text(helper.read_text().replace("'sglang': '0.5.22'", "'unknown': '0.5.22'"))
    with pytest.raises(SystemExit):
        module.check_engine_releases(root)
