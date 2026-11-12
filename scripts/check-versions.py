#!/usr/bin/env python3
"""Require one version across Rust, Python, release tooling and an optional tag."""

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path


def python_version(rust: str) -> str:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.(\d+))?", rust)
    if match is None:
        raise ValueError(f"unsupported Rust release version: {rust!r}")
    if match[2] is None:
        return match[1]
    return match[1] + {"alpha": "a", "beta": "b", "rc": "rc"}[match[2]] + match[3]


def read_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    return tomllib.loads(path.read_text())


def check_orbite_registry(root: Path) -> None:
    dependency = read_toml(root / "crates/orbitkv-core/Cargo.toml")["dependencies"][
        "orbite"
    ]
    version = dependency.get("version", "")
    if "git" in dependency or "path" in dependency or not version.startswith("="):
        raise ValueError(
            "publication blocked: first publish OrbiTE, then pin its exact crates.io "
            "version in orbitkv-core and rebuild; Git/path candidates cannot publish"
        )
    locked = [
        p for p in read_toml(root / "Cargo.lock")["package"] if p["name"] == "orbite"
    ]
    if (
        len(locked) != 1
        or locked[0]["version"] != version[1:]
        or locked[0].get("source")
        != "registry+https://github.com/rust-lang/crates.io-index"
        or re.fullmatch(r"[0-9a-f]{64}", locked[0].get("checksum", "")) is None
    ):
        raise ValueError(
            "publication blocked: OrbiTE lock entry needs the exact crates.io version and checksum"
        )


def check_versions(root: Path) -> tuple[str, str]:
    cargo = read_toml(root / "Cargo.toml")
    manifest = read_toml(root / "python/pyproject.toml")
    rust = cargo["workspace"]["package"]["version"]
    python = python_version(rust)
    if manifest["project"]["name"] != "orbitkv":
        raise ValueError("Python distribution name must be orbitkv")
    if manifest["project"]["version"] != python:
        raise ValueError(f"Python version must be {python} for Rust {rust}")
    if manifest["tool"]["commitizen"]["version"] != rust:
        raise ValueError("Commitizen version must match the Rust release/tag spelling")
    members = cargo["workspace"].get("members", [])
    if members:
        locked = read_toml(root / "Cargo.lock")["package"]
        for member in members:
            package = read_toml(root / member / "Cargo.toml")["package"]
            if (
                package.get("version") != {"workspace": True}
                or package.get("publish") is not False
            ):
                raise ValueError(
                    f"{member} must inherit its version and remain unpublished"
                )
            versions = [p["version"] for p in locked if p["name"] == package["name"]]
            if versions != [rust]:
                raise ValueError(f"{package['name']} lock version mismatch: {versions}")
    return rust, python


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
    manifest = read_toml(root / "python/pyproject.toml")
    extras = manifest.get("project", {}).get("optional-dependencies", {})
    for engine, version in versions.items():
        extra = extras.get(engine)
        requirement = f"{engine}=={version}"
        upper_python = "3.14" if engine == "sglang" else "3.15"
        requirement += (
            f"; python_version >= '3.11' and python_version < '{upper_python}'"
            " and sys_platform == 'linux'"
        )
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
        "--publication",
        action="store_true",
        help="Require registry-sourced OrbiTE before publishing",
    )
    parser.add_argument(
        "--format", choices=("python", "rust", "json"), default="python"
    )
    parser.add_argument(
        "--engines",
        action="store_true",
        help="Check engine extras, release gate and gitlinks",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.engines:
        check_engine_releases(root)
    try:
        rust, python = check_versions(root)
        if args.tag is not None and args.tag != f"v{rust}":
            raise ValueError(f"release tag {args.tag!r} must match v{rust}")
        if args.publication:
            check_orbite_registry(root)
    except (KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.format == "json":
        locked = read_toml(root / "Cargo.lock")["package"]
        orbite = [p for p in locked if p["name"] == "orbite"]
        print(
            json.dumps(
                {
                    "rust": rust,
                    "python": python,
                    "orbite": orbite,
                    "revision": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=root, text=True
                    ).strip(),
                },
                indent=2,
            )
        )
    else:
        print(python if args.format == "python" else rust)


if __name__ == "__main__":
    main()
