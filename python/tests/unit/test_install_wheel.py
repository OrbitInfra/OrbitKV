import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from tests.support.install_wheel import install


def member(name: str, mode: int, data: bytes = b"payload") -> ZipInfo:
    value = ZipInfo(name)
    value.create_system = 3
    value.external_attr = mode << 16
    return value


def wheel(tmp_path: Path, entries: list[tuple[ZipInfo, bytes]]) -> Path:
    path = tmp_path / "candidate.whl"
    with ZipFile(path, "w") as archive:
        for info, data in entries:
            archive.writestr(info, data)
    return path


def test_exact_wheel_install_preserves_manager_mode_and_rejects_reuse(tmp_path):
    candidate = wheel(
        tmp_path,
        [
            (member("orbitkv/__init__.py", stat.S_IFREG | 0o644), b""),
            (member("orbitkv/orbitkv-cache-manager-py", stat.S_IFREG | 0o755), b"manager"),
        ],
    )
    target = tmp_path / "installed"
    target.mkdir()
    result = install(candidate, target)
    manager = target / "orbitkv/orbitkv-cache-manager-py"
    assert result["count"] == 2
    assert stat.S_IMODE(manager.stat().st_mode) == 0o755
    assert manager.read_bytes() == b"manager"
    with pytest.raises(RuntimeError, match="empty"):
        install(candidate, target)


@pytest.mark.parametrize(
    "entry",
    [
        member("../escape", stat.S_IFREG | 0o644),
        member("/absolute", stat.S_IFREG | 0o644),
        member("orbitkv/manager", stat.S_IFLNK | 0o777),
        member("orbitkv/manager", stat.S_IFREG | stat.S_ISUID | 0o755),
    ],
)
def test_exact_wheel_install_rejects_unsafe_members(tmp_path, entry):
    candidate = wheel(tmp_path, [(entry, b"unsafe")])
    target = tmp_path / "installed"
    target.mkdir()
    with pytest.raises(RuntimeError):
        install(candidate, target)
    assert not any(target.iterdir())


def test_exact_wheel_install_rejects_non_unix_and_duplicate_members(tmp_path):
    non_unix = member("orbitkv/manager", stat.S_IFREG | 0o755)
    non_unix.create_system = 0
    target = tmp_path / "installed"
    target.mkdir()
    with pytest.raises(RuntimeError, match="Unix"):
        install(wheel(tmp_path, [(non_unix, b"manager")]), target)
    duplicate = member("orbitkv/manager", stat.S_IFREG | 0o755)
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate_wheel = wheel(tmp_path, [(duplicate, b"one"), (duplicate, b"two")])
    with pytest.raises(RuntimeError, match="duplicated"):
        install(duplicate_wheel, target)
