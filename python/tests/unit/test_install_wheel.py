import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from tests.support import install_wheel as installer
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
    result = install(candidate, target)
    manager = target / "orbitkv/orbitkv-cache-manager-py"
    assert result["count"] == 2
    assert stat.S_IMODE(manager.stat().st_mode) == 0o755
    assert manager.read_bytes() == b"manager"
    with pytest.raises(RuntimeError, match="must not exist"):
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
    with pytest.raises(RuntimeError):
        install(candidate, target)
    assert not target.exists()


def test_exact_wheel_install_rejects_non_unix_and_duplicate_members(tmp_path):
    non_unix = member("orbitkv/manager", stat.S_IFREG | 0o755)
    non_unix.create_system = 0
    target = tmp_path / "installed"
    with pytest.raises(RuntimeError, match="Unix"):
        install(wheel(tmp_path, [(non_unix, b"manager")]), target)
    duplicate = member("orbitkv/manager", stat.S_IFREG | 0o755)
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate_wheel = wheel(tmp_path, [(duplicate, b"one"), (duplicate, b"two")])
    with pytest.raises(RuntimeError, match="duplicated"):
        install(duplicate_wheel, target)
    assert not target.exists()


def test_exact_wheel_install_applies_directory_modes_after_children(tmp_path):
    candidate = wheel(
        tmp_path,
        [
            (member("private/payload", stat.S_IFREG | 0o600), b"payload"),
            (member("private/", stat.S_IFDIR | 0o700), b""),
        ],
    )
    target = tmp_path / "installed"
    install(candidate, target)
    assert stat.S_IMODE((target / "private").stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "entries",
    [
        [
            (member("pkg//payload", stat.S_IFREG | 0o600), b"one"),
            (member("pkg/payload", stat.S_IFREG | 0o600), b"two"),
        ],
        [
            (member("pkg", stat.S_IFREG | 0o600), b"file"),
            (member("pkg/payload", stat.S_IFREG | 0o600), b"child"),
        ],
        [
            (member("pkg/", stat.S_IFDIR | 0o700), b""),
            (member("pkg", stat.S_IFREG | 0o600), b"file"),
        ],
    ],
)
def test_exact_wheel_install_rejects_normalized_overlaps_before_writes(tmp_path, entries):
    candidate = wheel(tmp_path, entries)
    target = tmp_path / "installed"
    with pytest.raises(RuntimeError, match="duplicated|parent|both"):
        install(candidate, target)
    assert not target.exists()


def test_exact_wheel_install_does_not_follow_replaced_parent(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate = wheel(
        tmp_path,
        [(member("pkg/payload", stat.S_IFREG | 0o600), b"payload")],
    )
    target = workspace / "installed"
    original = ZipFile.open
    replaced = False

    def replace_parent(archive, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            retired = tmp_path / "retired-workspace"
            workspace.rename(retired)
            workspace.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original(archive, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", replace_parent)
    with pytest.raises(RuntimeError, match="parent|identity"):
        install(candidate, target)
    assert not (outside / "installed").exists()


def test_exact_wheel_install_rejects_preexisting_target(tmp_path):
    candidate = wheel(tmp_path, [(member("payload", stat.S_IFREG | 0o600), b"payload")])
    target = tmp_path / "installed"
    target.mkdir()
    with pytest.raises(RuntimeError, match="must not exist"):
        install(candidate, target)


def test_exact_wheel_install_rejects_replaced_staging_name(tmp_path, monkeypatch):
    candidate = wheel(
        tmp_path,
        [(member("pkg/payload", stat.S_IFREG | 0o600), b"payload")],
    )
    target = tmp_path / "installed"
    original = installer.write_member
    replaced = False

    def replace_stage(archive, wheel_member, parts, mode, root_fd):
        nonlocal replaced
        if not replaced:
            stage = Path(f"/proc/self/fd/{root_fd}").resolve(strict=True)
            retired = stage.with_name(stage.name + "-retained")
            stage.rename(retired)
            stage.mkdir()
            (stage / "injected").write_bytes(b"not the wheel")
            replaced = True
        return original(archive, wheel_member, parts, mode, root_fd)

    monkeypatch.setattr(installer, "write_member", replace_stage)
    with pytest.raises(RuntimeError, match="staging.*identity"):
        install(candidate, target)
    assert not target.exists()


def test_exact_wheel_install_preserves_primary_post_publish_error(tmp_path, monkeypatch):
    candidate = wheel(tmp_path, [(member("payload", stat.S_IFREG | 0o600), b"payload")])
    target = tmp_path / "installed"
    original = installer.rename_noreplace

    def move_published_target(parent_fd, source, name):
        original(parent_fd, source, name)
        parent = Path(f"/proc/self/fd/{parent_fd}").resolve(strict=True)
        (parent / name).rename(parent / "retired-installation")

    monkeypatch.setattr(installer, "rename_noreplace", move_published_target)
    with pytest.raises(FileNotFoundError):
        install(candidate, target)
    assert not target.exists()
    assert (tmp_path / "retired-installation").is_dir()
