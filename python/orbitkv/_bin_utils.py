"""Utility to locate Rust binaries bundled with orbitkv."""

import os
import shutil
import site
import sysconfig
from pathlib import Path

# orbitkv/ directory (where this file lives)
_MODULE_DIR = Path(__file__).parent
# Repo root: orbitkv/python/orbitkv/../../ -> orbitkv/
_REPO_ROOT = _MODULE_DIR.parent.parent


def find_binary(name: str) -> str:
    """Locate an OrbitKV binary by name.

    Search order:
    1. Installed package directory (pip install from wheel)
    2. Cargo target/release/ (source checkout)
    3. Cargo target/debug/
    4. PATH fallback
    """
    # A wheel must execute its own binary, even inside a source checkout.
    path = _MODULE_DIR / name
    if path.is_file():
        return str(path)

    # Dev mode: cargo target/release/
    path = _REPO_ROOT / "target" / "release" / name
    if path.is_file():
        return str(path)

    # Dev mode: cargo target/debug/
    path = _REPO_ROOT / "target" / "debug" / name
    if path.is_file():
        return str(path)

    # Fallback: PATH
    found = shutil.which(name)
    if found:
        return found

    return name


def binary_env() -> dict[str, str]:
    """Return an environment that can load binaries linked to this Python."""
    env = os.environ.copy()
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            path for path in (env.get("LD_LIBRARY_PATH", ""), libdir) if path
        )
    paths = [
        sysconfig.get_path("stdlib"),
        sysconfig.get_path("platstdlib"),
        sysconfig.get_config_var("DESTSHARED"),
        str(_MODULE_DIR.parent),
        *env.get("PYTHONPATH", "").split(os.pathsep),
        *site.getsitepackages(),
    ]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    return env
