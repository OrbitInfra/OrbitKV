"""Install an exact qualification wheel without discarding Unix file modes."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import os
import secrets
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

MAX_FILES = 4096
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
RENAME_NOREPLACE = 1


def member_spec(member: ZipInfo) -> tuple[tuple[str, ...], int, bool]:
    if member.create_system != 3 or "\\" in member.filename or "\x00" in member.filename:
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
    directory = member.is_dir()
    if directory:
        if kind not in (0, stat.S_IFDIR):
            raise RuntimeError(f"Wheel directory type differs: {member.filename}")
    elif kind != stat.S_IFREG:
        raise RuntimeError(f"Wheel member is not a regular file: {member.filename}")
    if mode & 0o7000:
        raise RuntimeError(f"Wheel member has privileged mode bits: {member.filename}")
    return relative.parts, mode & 0o777, directory


def archive_plan(archive: ZipFile) -> tuple[list[tuple[ZipInfo, tuple[str, ...], int]], dict]:
    members = archive.infolist()
    if not members or len(members) > MAX_FILES:
        raise RuntimeError("Wheel member count is invalid")
    files = []
    file_paths = set()
    directory_modes = {}
    normalized = set()
    total = 0
    for member in members:
        parts, mode, directory = member_spec(member)
        if parts in normalized:
            raise RuntimeError(f"Wheel member is duplicated after normalization: {member.filename}")
        normalized.add(parts)
        total += member.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise RuntimeError("Wheel uncompressed bytes exceed the bound")
        if directory:
            directory_modes[parts] = mode or 0o755
        else:
            file_paths.add(parts)
            files.append((member, parts, mode))
    for path in normalized:
        for length in range(1, len(path)):
            if path[:length] in file_paths:
                raise RuntimeError(f"Wheel file is also a parent: {'/'.join(path[:length])}")
    if file_paths & set(directory_modes):
        raise RuntimeError("Wheel path is both a file and directory")
    directories = set(directory_modes)
    for path in normalized:
        directories.update(path[:length] for length in range(1, len(path)))
    return files, {
        "directories": directories,
        "directory_modes": directory_modes,
        "member_count": len(members),
        "bytes": total,
    }


def open_directory(root_fd: int, parts: tuple[str, ...]) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in parts:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def create_directories(root_fd: int, directories: set[tuple[str, ...]]) -> None:
    for parts in sorted(directories, key=lambda value: (len(value), value)):
        parent = open_directory(root_fd, parts[:-1])
        try:
            with contextlib.suppress(FileExistsError):
                os.mkdir(parts[-1], 0o700, dir_fd=parent)
            child = os.open(
                parts[-1],
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            os.close(child)
            os.fsync(parent)
        finally:
            os.close(parent)


def write_member(
    archive: ZipFile,
    member: ZipInfo,
    parts: tuple[str, ...],
    mode: int,
    root_fd: int,
) -> None:
    parent = open_directory(root_fd, parts[:-1])
    descriptor = None
    try:
        descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        copied = 0
        with archive.open(member) as source:
            while chunk := source.read(1024 * 1024):
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise RuntimeError(f"Wheel member write failed: {member.filename}")
                    copied += written
                    view = view[written:]
        if copied != member.file_size or os.fstat(descriptor).st_size != member.file_size:
            raise RuntimeError(f"Wheel member size differs: {member.filename}")
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != mode:
            raise RuntimeError(f"Wheel member mode differs: {member.filename}")
        os.fsync(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def finalize_directories(
    root_fd: int,
    directories: set[tuple[str, ...]],
    modes: dict[tuple[str, ...], int],
) -> None:
    for parts in sorted(directories, key=lambda value: (-len(value), value)):
        descriptor = open_directory(root_fd, parts)
        try:
            mode = modes.get(parts, 0o755)
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != mode:
                raise RuntimeError(f"Wheel directory mode differs: {'/'.join(parts)}")
        finally:
            os.close(descriptor)
    os.fchmod(root_fd, 0o755)
    os.fsync(root_fd)


def same_identity(path: Path, descriptor: int) -> bool:
    current = path.stat()
    opened = os.fstat(descriptor)
    return (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)


def rename_noreplace(parent_fd: int, source: str, target: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            parent_fd,
            os.fsencode(source),
            parent_fd,
            os.fsencode(target),
            RENAME_NOREPLACE,
        )
        == 0
    ):
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise RuntimeError("Installation target appeared before publication")
    raise OSError(error, os.strerror(error))


def install(wheel: Path, target: Path) -> dict:
    if (
        not wheel.is_absolute()
        or wheel.resolve(strict=True) != wheel
        or not stat.S_ISREG(wheel.lstat().st_mode)
    ):
        raise RuntimeError("Wheel must be one canonical regular file")
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise RuntimeError("Installation target must not exist")
    parent_path = target.parent
    if parent_path.resolve(strict=True) != parent_path or not parent_path.is_dir():
        raise RuntimeError("Installation parent must be one canonical directory")
    with ZipFile(wheel) as archive:
        files, plan = archive_plan(archive)
        parent_fd = os.open(
            parent_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        stage = f".orbitkv-install-{target.name}-{os.getpid()}-{secrets.token_hex(8)}"
        root_fd = None
        published = False
        try:
            os.mkdir(stage, 0o700, dir_fd=parent_fd)
            root_fd = os.open(
                stage,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            create_directories(root_fd, plan["directories"])
            installed = []
            for member, parts, mode in files:
                write_member(archive, member, parts, mode, root_fd)
                installed.append(
                    {"path": str(target.joinpath(*parts)), "bytes": member.file_size, "mode": mode}
                )
            finalize_directories(
                root_fd,
                plan["directories"],
                plan["directory_modes"],
            )
            if not same_identity(parent_path, parent_fd):
                raise RuntimeError("Installation parent changed before publication")
            rename_noreplace(parent_fd, stage, target.name)
            published = True
            os.fsync(parent_fd)
            if (
                not same_identity(parent_path, parent_fd)
                or target.resolve(strict=True) != target
                or not stat.S_ISDIR(target.lstat().st_mode)
            ):
                raise RuntimeError("Published installation identity differs")
            return {"files": installed, "count": len(installed), "bytes": plan["bytes"]}
        finally:
            if root_fd is not None:
                os.close(root_fd)
            os.close(parent_fd)
            if published and not target.exists():
                raise RuntimeError("Published installation disappeared")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    result = install(args.wheel, args.target)
    print({key: result[key] for key in ("count", "bytes")})


if __name__ == "__main__":
    main()
