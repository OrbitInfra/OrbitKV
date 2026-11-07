"""Repair all bundled ELF dependencies and retain their redistribution notices."""

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from auditwheel.wheel_abi import analyze_wheel_abi
from wheel.wheelfile import WheelFile

# These are supplied by the matching Python/CUDA/RDMA installation. In particular,
# auditwheel must not remove libpython from the standalone embedded-Python Manager.
HOST_LIBRARIES = (
    "libpython3.*.so*",
    "libcuda.so*",
    "libcudart.so*",
    "libnvrtc*.so*",
    "libcufile*.so*",
    "libibverbs.so*",
    "librdmacm.so*",
    "libmlx5.so*",
)


def library_notice(source: Path) -> Path:
    # A prebuilt runtime must provide notices for libraries outside the OS package database.
    adjacent = source.with_name(source.name + ".license")
    if adjacent.is_file():
        return adjacent
    candidates = {str(source), str(source.resolve())}
    candidates.update(
        path.removeprefix("/usr")
        for path in tuple(candidates)
        if path.startswith("/usr/lib/")
    )
    candidates.update(
        "/usr" + path for path in tuple(candidates) if path.startswith("/lib/")
    )
    for path in sorted(candidates):
        result = subprocess.run(
            ["dpkg-query", "-S", path], capture_output=True, text=True, check=False
        )
        if result.returncode:
            continue
        for line in result.stdout.splitlines():
            package = line.split(": ", 1)[0].split(":", 1)[0]
            notice = Path("/usr/share/doc") / package / "copyright"
            if notice.is_file():
                return notice
    raise ValueError(f"No redistribution notice for {source}; supply {adjacent}")


def repair(source: Path, output: Path) -> Path:
    abi = analyze_wheel_abi(
        None,
        None,
        source,
        frozenset(HOST_LIBRARIES),
        disable_isa_ext_check=False,
        allow_graft=True,
    )
    dependencies = {
        path
        for policies in abi.full_external_refs.values()
        for reference in policies.values()
        for path in reference.libs.values()
        if path is not None
    }
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.whl")):
        raise ValueError("repair output must not contain an earlier wheel")
    command = [
        sys.executable,
        "-m",
        "auditwheel",
        "repair",
        str(source),
        "--wheel-dir",
        str(output),
    ]
    for pattern in HOST_LIBRARIES:
        command.extend(("--exclude", pattern))
    subprocess.run(command, check=True)
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one repaired wheel")
    repaired = wheels[0]
    with ZipFile(repaired) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        dist_info = metadata_path.rsplit("/", 1)[0]
        grafted = {
            Path(name).name
            for name in archive.namelist()
            if ".libs/" in name and not name.endswith("/")
        }
        notices = {}
        for dependency in sorted(dependencies):
            digest = hashlib.sha256(dependency.read_bytes()).hexdigest()[:8]
            base, suffix = dependency.name.split(".", 1)
            name = f"{base}-{digest}.{suffix}"
            if name not in grafted:
                continue
            grafted.remove(name)
            notice = library_notice(dependency).read_bytes()
            notices[f"native/{name}.txt"] = notice
            for common in re.findall(
                rb"/usr/share/common-licenses/([A-Za-z0-9.+-]+)", notice
            ):
                common_name = common.decode().rstrip(".")
                notices[f"native/common/{common_name}"] = (
                    Path("/usr/share/common-licenses") / common_name
                ).read_bytes()
        if grafted:
            raise ValueError(f"Missing dependency provenance for {sorted(grafted)}")
        metadata = archive.read(metadata_path).decode()
        headers, separator, body = metadata.partition("\n\n")
        if not separator:
            raise ValueError("invalid wheel metadata")
        metadata = (
            headers
            + "\n"
            + "".join(f"License-File: {name}\n" for name in sorted(notices))
            + "\n"
            + body
        )
        with tempfile.TemporaryDirectory(dir=output) as directory:
            staged = Path(directory) / repaired.name
            with WheelFile(staged, "w") as wheel:
                for entry in archive.infolist():
                    if entry.filename == f"{dist_info}/RECORD":
                        continue
                    wheel.writestr(
                        entry,
                        metadata.encode()
                        if entry.filename == metadata_path
                        else archive.read(entry),
                    )
                for name, contents in sorted(notices.items()):
                    wheel.writestr(f"{dist_info}/licenses/{name}", contents)
            # The input archive remains open until every entry and notice is copied.
            staged.replace(repaired)
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(
        f"Repaired wheel with native notices: {repair(args.wheel.resolve(), args.output.resolve())}"
    )


if __name__ == "__main__":
    main()
