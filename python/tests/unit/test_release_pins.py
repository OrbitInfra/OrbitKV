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
    versions = {"vllm": "0.33.0", "sglang": "0.5.23"}
    commits = {"vllm": "a" * 40, "sglang": "b" * 40}
    requirements = {
        name: (
            f"{name}=={version}; python_version >= '3.11' "
            f"and python_version < '{'3.14' if name == 'sglang' else '3.15'}' "
            "and sys_platform == 'linux'"
        )
        for name, version in versions.items()
    }
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
            "vllm = [\n  \"vllm==0.33.0; python_version >= '3.11' and python_version < '3.15' and sys_platform == 'linux'\", # official release\n]\n"
            "sglang = [\"sglang==0.5.23; python_version >= '3.11' and python_version < '3.14' and sys_platform == 'linux'\"]\n"
        )
    module.check_engine_releases(root)


@pytest.mark.parametrize("name", ["orbitkv", "orbitkv" + "-llm"])
def test_distribution_name_is_part_of_version_consistency(release_checker, monkeypatch, name):
    module, root, _, _ = release_checker
    (root / "Cargo.toml").write_text('[workspace.package]\nversion = "0.1.0"\n')
    (root / "python/pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\n[tool.commitizen]\nversion = "0.1.0"\n'
    )
    monkeypatch.setattr(module, "__file__", str(root / "scripts/check-versions.py"))
    monkeypatch.setattr(module.sys, "argv", ["check-versions.py"])
    if name == "orbitkv":
        module.main()
    else:
        with pytest.raises(SystemExit, match="distribution name must be orbitkv"):
            module.main()


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
        path.write_text(path.read_text().replace("vllm==0.33.0", "vllm==0.30.0"))
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
        helper.write_text(helper.read_text().replace("'sglang': '0.5.23'", "'unknown': '0.5.23'"))
    with pytest.raises(SystemExit):
        module.check_engine_releases(root)
