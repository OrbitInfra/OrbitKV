"""Release upgrades must align installable extras, gate inputs and gitlinks."""

import importlib.util
import json
import sys
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
    monkeypatch.setattr(sys, "argv", ["check-versions.py"])
    if name == "orbitkv":
        module.main()
    else:
        with pytest.raises(SystemExit, match="distribution name must be orbitkv"):
            module.main()


@pytest.mark.parametrize(
    "rust,python",
    [
        ("0.1.0-alpha.1", "0.1.0a1"),
        ("0.1.0-beta.1", "0.1.0b1"),
        ("0.1.0-rc.2", "0.1.0rc2"),
        ("0.1.0", "0.1.0"),
    ],
)
def test_rust_prerelease_has_the_canonical_python_spelling(release_checker, rust, python):
    module, _, _, _ = release_checker
    assert module.python_version(rust) == python
    with pytest.raises(ValueError, match="unsupported Rust"):
        module.python_version("0.1.0b1")


@pytest.fixture
def beta_candidate(release_checker):
    module, root, _, _ = release_checker
    (root / "Cargo.toml").write_text(
        '[workspace]\nmembers = ["crates/core"]\n[workspace.package]\nversion = "0.1.0-beta.1"\n'
    )
    (root / "crates/core").mkdir(parents=True)
    (root / "crates/core/Cargo.toml").write_text(
        '[package]\nname = "orbitkv-core"\nversion.workspace = true\npublish = false\n'
    )
    (root / "python/pyproject.toml").write_text(
        '[project]\nname = "orbitkv"\nversion = "0.1.0b1"\n'
        '[tool.commitizen]\nversion = "0.1.0-beta.1"\n'
    )
    (root / "Cargo.lock").write_text(
        '[[package]]\nname = "orbitkv-core"\nversion = "0.1.0-beta.1"\n'
    )
    return module, root


@pytest.mark.parametrize("drift", ["none", "python", "commitizen", "lock", "publish"])
def test_beta_versions_lock_and_private_crates_are_checked(beta_candidate, drift):
    module, root = beta_candidate
    if drift in ("python", "commitizen"):
        path = root / "python/pyproject.toml"
        old = 'version = "0.1.0b1"' if drift == "python" else 'version = "0.1.0-beta.1"'
        path.write_text(path.read_text().replace(old, 'version = "0.1.0"'))
    elif drift == "lock":
        path = root / "Cargo.lock"
        path.write_text(path.read_text().replace("0.1.0-beta.1", "0.1.0"))
    elif drift == "publish":
        path = root / "crates/core/Cargo.toml"
        path.write_text(path.read_text().replace("publish = false", "publish = true"))
    if drift == "none":
        assert module.check_versions(root) == ("0.1.0-beta.1", "0.1.0b1")
    else:
        with pytest.raises(ValueError):
            module.check_versions(root)


@pytest.mark.parametrize("tag", ["v0.1.0-beta.1", "v0.1.0", "v0.1.0b1", "v0.1.0-beta.2"])
def test_tag_uses_the_exact_rust_version(beta_candidate, monkeypatch, capsys, tag):
    module, root = beta_candidate
    monkeypatch.setattr(module, "__file__", str(root / "scripts/check-versions.py"))
    monkeypatch.setattr(sys, "argv", ["check-versions.py", "--tag", tag])
    if tag == "v0.1.0-beta.1":
        module.main()
        assert capsys.readouterr().out.strip() == "0.1.0b1"
    else:
        with pytest.raises(SystemExit, match="must match"):
            module.main()


@pytest.mark.parametrize(
    "source", ["registry", "git", "path", "unlocked", "wrong-version", "checksum"]
)
def test_publication_requires_exact_registry_orbite(beta_candidate, source):
    module, root = beta_candidate
    core = root / "crates/orbitkv-core"
    core.mkdir(parents=True)
    fields = 'version = "=0.1.0-beta.1"'
    if source == "git":
        fields += ', git = "https://github.com/OrbitInfra/OrbiTE"'
    elif source == "path":
        fields += ', path = "../orbite"'
    (core / "Cargo.toml").write_text(f"[dependencies]\norbite = {{ {fields} }}\n")
    lock = root / "Cargo.lock"
    if source != "unlocked":
        version = "0.1.0-alpha.1" if source == "wrong-version" else "0.1.0-beta.1"
        checksum = "invalid" if source == "checksum" else "a" * 64
        lock.write_text(
            lock.read_text() + '\n[[package]]\nname = "orbite"\n'
            f'version = "{version}"\nchecksum = "{checksum}"\n'
            'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        )
    if source == "registry":
        module.check_orbite_registry(root)
    else:
        with pytest.raises(ValueError, match="publication blocked"):
            module.check_orbite_registry(root)


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
