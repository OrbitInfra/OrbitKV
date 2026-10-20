#!/usr/bin/env python3
"""Require one version across Rust, Python, release tooling and an optional tag."""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


def check_engine_releases(root: Path) -> None:
    helper = root / "python/tests/support/installed_artifacts.py"
    constants = {}
    for node in ast.parse(helper.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "ENGINE_VERSIONS",
                    "ENGINE_COMMITS",
                }:
                    constants[target.id] = ast.literal_eval(node.value)
    versions = constants.get("ENGINE_VERSIONS", {})
    commits = constants.get("ENGINE_COMMITS", {})
    if set(versions) != {"vllm", "sglang"} or set(commits) != set(versions):
        raise SystemExit(
            "engine release gate must declare versions and commits for both engines"
        )
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    manifest = tomllib.loads((root / "python/pyproject.toml").read_text())
    extras = manifest.get("project", {}).get("optional-dependencies", {})
    for engine, version in versions.items():
        extra = extras.get(engine)
        requirement = f"{engine}=={version}"
        if engine == "sglang":
            requirement += "; python_version >= '3.11'"
        if extra != [requirement]:
            raise SystemExit(
                f"{engine} extra {extra!r} differs from release gate {requirement!r}"
            )
        commit = commits[engine]
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise SystemExit(f"invalid {engine} release commit: {commit!r}")
        entry = subprocess.check_output(
            ["git", "ls-files", "--stage", f"third-party/{engine}"], cwd=root, text=True
        ).split()
        if (
            len(entry) != 4
            or entry[0] != "160000"
            or entry[1] != commit
            or entry[2] != "0"
        ):
            raise SystemExit(
                f"{engine} gitlink differs from release gate {commit}: {entry}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Release tag, including the v prefix")
    parser.add_argument(
        "--engines",
        action="store_true",
        help="Check engine extras, release gate and gitlinks",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.engines:
        check_engine_releases(root)
    versions = {}
    for path, section in (
        ("Cargo.toml", "workspace.package"),
        ("python/pyproject.toml", "project"),
        ("python/pyproject.toml", "tool.commitizen"),
    ):
        source = (root / path).read_text()
        table = re.search(rf"(?ms)^\[{re.escape(section)}\]\n(.*?)(?:^\[|\Z)", source)
        version = (
            re.search(r'^version = "([^"]+)"$', table[1], re.MULTILINE)
            if table
            else None
        )
        if version is None:
            raise SystemExit(f"missing version in {path} [{section}]")
        versions[f"{path} [{section}]"] = version[1]
    if len(set(versions.values())) != 1:
        raise SystemExit(f"version mismatch: {versions}")
    version = next(iter(versions.values()))
    if args.tag is not None and args.tag != f"v{version}":
        raise SystemExit(f"release tag {args.tag!r} must match v{version}")
    print(version)


if __name__ == "__main__":
    main()
