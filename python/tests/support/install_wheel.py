"""Install an exact qualification wheel without discarding Unix file modes."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

MAX_FILES = 4096
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3


def member_path(target: Path, member: ZipInfo) -> tuple[Path, int]:
    if member.create_system != 3 or "\\" in member.filename:
        raise RuntimeError(f"Wheel member has no Unix identity: {member.filename}")
    relative = PurePosixPath(member.filename)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise RuntimeError(f"Wheel member leaves the installation: {member.filename}")
    mode = member.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if member.is_dir():
        if kind not in (0, stat.S_IFDIR):
            raise RuntimeError(f"Wheel directory type differs: {member.filename}")
    elif kind != stat.S_IFREG:
        raise RuntimeError(f"Wheel member is not a regular file: {member.filename}")
    if mode & 0o7000:
        raise RuntimeError(f"Wheel member has privileged mode bits: {member.filename}")
    return target.joinpath(*relative.parts), mode & 0o777


def make_parents(target: Path, parent: Path) -> None:
    missing = []
    current = parent
    while current != target and not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_relative_to(target):
        raise RuntimeError(f"Wheel parent leaves the installation: {parent}")
    for directory in reversed(missing):
        directory.mkdir(mode=0o755)
    current = parent
    while True:
        metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or current.resolve(strict=True) != current:
            raise RuntimeError(f"Wheel parent is not a canonical directory: {current}")
        if current == target:
            break
        current = current.parent


def install(wheel: Path, target: Path) -> dict:
    if not wheel.is_absolute() or wheel.resolve(strict=True) != wheel or not wheel.is_file():
        raise RuntimeError("Wheel must be one canonical regular file")
    if not target.is_absolute() or target.resolve(strict=True) != target or not target.is_dir():
        raise RuntimeError("Installation target must be one canonical directory")
    if any(target.iterdir()):
        raise RuntimeError("Installation target must be empty")
    installed = []
    total = 0
    names = set()
    with ZipFile(wheel) as archive:
        members = archive.infolist()
        if not members or len(members) > MAX_FILES:
            raise RuntimeError("Wheel member count is invalid")
        prepared = []
        for member in members:
            if member.filename in names:
                raise RuntimeError(f"Wheel member is duplicated: {member.filename}")
            names.add(member.filename)
            total += member.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise RuntimeError("Wheel uncompressed bytes exceed the bound")
            path, mode = member_path(target, member)
            prepared.append((member, path, mode))
        for member, path, mode in prepared:
            make_parents(target, path.parent)
            if member.is_dir():
                path.mkdir(mode=mode or 0o755, exist_ok=True)
                if path.resolve(strict=True) != path:
                    raise RuntimeError(f"Wheel directory is not canonical: {path}")
                continue
            if path.exists() or path.is_symlink():
                raise RuntimeError(f"Wheel member already exists: {path}")
            with archive.open(member) as source, path.open("xb") as destination:
                copied = shutil.copyfileobj(source, destination, length=1024 * 1024)
                if copied is not None:
                    raise RuntimeError("Unexpected copy result")
                destination.flush()
                os.fsync(destination.fileno())
            if path.stat().st_size != member.file_size:
                raise RuntimeError(f"Wheel member size differs: {path}")
            path.chmod(mode)
            installed.append({"path": str(path), "bytes": member.file_size, "mode": mode})
    descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"files": installed, "count": len(installed), "bytes": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    result = install(args.wheel, args.target)
    print({key: result[key] for key in ("count", "bytes")})


if __name__ == "__main__":
    main()
