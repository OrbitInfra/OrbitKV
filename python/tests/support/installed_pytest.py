"""Run installed-wheel checks and pytest through the same isolated entry point."""

import argparse
import importlib
import importlib.machinery
import importlib.metadata
import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("installed-wheel qualification requires Python without -O or -OO")
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    if args.probe == bool(pytest_args):
        parser.error("choose --probe or pytest arguments after --")
    for path in (args.installation, args.harness):
        if not path.is_absolute() or path.resolve(strict=True) != path or not path.is_dir():
            parser.error("installation and harness must be canonical absolute directories")
    if any(name == "orbitkv" or name.startswith("orbitkv.") for name in sys.modules):
        raise RuntimeError("OrbitKV was imported before selecting the frozen installation")

    sys.path[:0] = [
        str(args.installation),
        str(args.harness / "python"),
        str(args.harness),
    ]
    package = importlib.import_module("orbitkv")
    extension = importlib.import_module("orbitkv.orbitkv")
    for module in (package, extension):
        if Path(module.__file__).resolve().parent != args.installation / "orbitkv":
            raise RuntimeError("Installed package origin is outside the frozen wheel")
    if not any(
        extension.__file__.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ):
        raise RuntimeError("The selected OrbitKV module is not a native extension")

    from tests.support import installed_artifacts

    guard_path = args.harness / "python/tests/support/installed_artifacts.py"
    if Path(installed_artifacts.__file__).resolve() != guard_path:
        raise RuntimeError("Installed-artifact checks are outside the frozen harness")
    distributions = [
        distribution
        for distribution in importlib.metadata.distributions(path=[str(args.installation)])
        if distribution.metadata["Name"] == "orbitkv-llm"
    ]
    if len(distributions) != 1:
        raise RuntimeError("The installation must contain exactly one OrbitKV distribution")
    snapshot = installed_artifacts.distribution_snapshot(distributions[0], "orbitkv")
    print(
        json.dumps(
            {
                "orbitkv_file": package.__file__,
                "extension": extension.__file__,
                "sys_path": sys.path,
                "installed_artifact_guard": str(guard_path),
                "record_files_checked": len(snapshot["files"]),
                "installed_executables": snapshot["executables"],
                "probe": args.probe,
            }
        ),
        flush=True,
    )
    if args.probe:
        return
    sys.argv = ["pytest", *pytest_args]
    runpy.run_module("pytest", run_name="__main__")


if __name__ == "__main__":
    main()
